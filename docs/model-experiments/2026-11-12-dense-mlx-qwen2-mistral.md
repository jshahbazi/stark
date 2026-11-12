# 20261112-DENSE-MLX-01: MLX affine import for Qwen2 and Mistral dense

The canonical source-bound record is the
[dated run record](../archive/runs/2026-11-12-dense-mlx-qwen2-mistral.md). Index:
[ledger](README.md).

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-DENSE-MLX-01`; registered and executed 2026-11-12; Claude agent session on branch `claude/dense-mlx-qwen2-mistral` |
| Kind | Compatibility and precision (new weight-precision import path for two existing families) |
| Attempt state | Completed for structural scope; numerical, performance and artifact lifecycle scope blocked on approval for weight downloads |
| Result and acceptance criterion | Structural: passed. All nine sampled repositories pass the real `preflight_model` and `plan_from_headers` from their pinned headers, and the model-free planner contracts pass. Numerical parity, speed and prefix reuse on these artifacts: resource-deferred (no weight download or inference was approved) |
| Evidence scope | Source review of the native engine; model-free converter tests; live header-only planning. No native load, no full-model execution |
| Promotion decision | Awaiting qualification. The catalogue now offers these rows for download; no artifact is qualified |
| Disposition | Adopted on branch `claude/dense-mlx-qwen2-mistral` (base `32d1582b`), kept for owner review; not pushed, merged, released or installed |
| Predecessor / related attempts / governing issue / PR | Mirrors the Llama MLX importer added in `d7ab8dd3` (runtime qualification also separate there). Related: [20261110-CATALOG-01](2026-11-10-catalog-family-admission.md). No PR |

## Hypothesis and change

- Concrete problem, trigger and expected behavior: Qwen2 and Mistral MLX
  affine exports were reported as "This family and mode do not implement the
  declared quantized representation" in **Model Catalog**, although the shared
  dense planner and the MLX repacking helpers are family-independent. Expected:
  uniform 4-bit and 8-bit MLX exports of these families plan exactly as Llama
  exports do, with Qwen2 attention biases kept as separate vectors.
- Baseline and candidate difference: `adapters/dense.py` parsed MLX
  quantization and accepted packed matrices only when `family == "llama"`, and
  raised "Quantized checkpoint is outside qualified dense coverage" otherwise;
  `model_support._MLX` omitted `qwen2` and `mistral`. The candidate removes the
  three family gates and adds both families to `_MLX`. Bit widths, group sizes,
  per-module override rejection and every other family are unchanged.
- Correctness metrics, fixed thresholds, controls and references set before the
  run: structural only. Every sampled repository must pass the unmodified
  preflight and header planner with zero full-precision matrix specs; the
  model-free suite must have no new failures against the recorded baseline
  failing set.
- Qualification artifacts versus implementation sources: implementation is the
  shared dense planner. No reference outputs were generated.
- Why stopped: weight downloads and inference were outside the approved scope
  of this attempt.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | Same nine repositories, rejected by preflight | Nine repositories at the revisions in the run record (eight Qwen2.5/QwQ, one Mistral 7B v0.3) |
| Format/quantization and config/weight/tokenizer/processor/template hashes | MLX affine 4-bit and 8-bit, group 64 | Config SHA-256 and header-inventory SHA-256 per repository in the run record; weight, tokenizer and template hashes not recorded (not downloaded) |
| Backend, vendor revision and local patch provenance | Native Metal dense decoder, unchanged | Unchanged; no Rust change |
| Stark commit, dirty patch/runtime manifest and executable hash | `32d1582b` | Branch `claude/dense-mlx-qwen2-mistral`; no executable built |
| Toolchains, dependency lock hashes and build flags | `uv` Python 3.13.13 | Same; `tools/converter/uv.lock` SHA-256 `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Weight, activation, accumulation, recurrent-state and KV precision | Not applicable | Weights: unsigned affine 4-bit or 8-bit codes repacked byte-exactly. Scale planes are F16 in every sample, and the planner requires the affine bias planes to share that dtype; the default BF16 output rounds both to BF16, an F32 output keeps them exactly. Qwen2 attention-bias vectors are cast to the output dtype; their source dtype was not recorded. Activations and KV cache: the selected output dtype. No recurrent state |
| Selected kernel and materialization policies | Not applicable | Expected generic quantized GEMV/GEMM with fused output bias; not observed |
| Hardware, physical RAM, OS/driver/Metal toolchain | Not applicable | Darwin 25.6.0 host for converter tests only; no Metal execution |
| Available headroom, observed peak memory and measurement method | Not measured | Not measured |

## Workload and controls

- Image capability: N/A. Every sample declares `Qwen2ForCausalLM` or
  `MistralForCausalLM`, and the planner rejects unknown config keys, so none
  carries a `vision_config`. These are text-only sources; text gates remain.
- Cache exactness on the default image-loaded worker: N/A (text-only).
- Matched text-only versus image-loaded performance: N/A (text-only).
- MTP inventory: the planner requires the tensor set to equal the expected
  dense inventory, and all nine passed, so no draft-head tensors exist in these
  exports. No compatible MTP path; recorded as actual absence.
- Text MTP/image transitions: N/A.
- Task, budgets, sampling, cold/warm, repeats: not run.
- Network: Hugging Face only, unauthenticated, config, shard index and tensor
  headers only, at least 2.5 seconds between repositories.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Prior in-memory simulation | Planner with the three gates removed, real headers | All nine planned | Diagnostic; not the shipped code |
| Engine source review | Load path for MLX matrices with biases, sliding window, untied readout, prefix reuse | Supported by existing code; see run record | Code reading only |
| Model-free tests | New Qwen2/Mistral MLX contracts and refusal cases | Passed | Valid structural evidence |
| Live verification | Real `preflight_model` and `plan_from_headers` on nine repositories | 9/9 conditional and planned; 0 full-precision specs | Valid structural evidence for those revisions |
| Native load, parity, speed, prefix reuse | Not attempted | Not run | Resource-deferred |

## Correctness and lifecycle

- Reference and tolerance: none run. Numerical match against each full-precision
  parent is not established.
- Model-free versus artifact execution: model-free and header-only only.
- Prefix reuse: implemented by code, artifact qualification pending. The
  attention mixer allows rewind when it owns KV heads and has no ring
  (`crates/stark-engine/src/encodable_block/mixer/attention/mod.rs`,
  `prefix_reuse_supported`), and snapshots whenever it owns KV heads. Neither
  test reads the weight specification, and the KV cache uses the activation
  dtype. The worker key binds the loaded model instance, so a quantized
  artifact never shares a prefix with its parent. A Mistral or Qwen2 export with
  an active sliding window uses the ring cache and the snapshot route, as its
  full-precision export does. No sampled export activates a window.
- Rejection, cancellation and recovery: unchanged shared worker behavior; not
  exercised on these artifacts.
- Tests actually run: see the run record.

## Performance and memory

| Metric and measurement boundary | Baseline | Candidate | Repeats/dispersion | Eligibility and conclusion |
| --- | --- | --- | --- | --- |
| Tokenization/rendering/planning | Not measured | Not measured | None | None |
| First token / prefill | Not measured | Not measured | None | None |
| Useful decode / end-to-end rate | Not measured | Not measured | None | None |
| Observed peak memory / retained payload | Not measured | Not measured | None | None |

## Evidence, decision and limits

- Public aggregates: per-repository identities and planning results in the run
  record; private receipt `verify_dense_mlx.jsonl` SHA-256
  `3f7684b9b9cbc75e8d5ee7cb5b7a21c486a27226b6b28edb036003e5e9ded4c8`, produced
  by a harness with SHA-256
  `aa4cabff3c74c45959df7a162092fe173962310dd064d42f5dda3eefc4a5ab27` (both kept
  outside the repository).
- What worked: structural import planning for the nine revisions and the
  model-free contracts.
- Adoption: branch only, for owner review.
- Untested: native load of any converted artifact; numerical match with the
  BF16 parent; decode and prefill speed; prefix reuse exactness; tool-call
  behavior; tied-embedding exports (all nine are untied); active sliding
  windows (none sampled); other repositories and later revisions.
- Next action: with approval for a download, convert one small export per
  family (for example a Qwen2.5 0.5B or 1.5B MLX 4-bit export with tied
  embeddings, and Mistral 7B v0.3 4-bit), record a native loader
  qualification, compare greedy outputs and logits with the BF16 parent under a
  preregistered tolerance, run the prefix-cache validator, and measure speed
  against the BF16 conversion.

Keep prompts, completions, token vectors, probabilities, credentials, exception
dumps and model/private evaluation files out of this document.

## Follow-up history

None yet.
