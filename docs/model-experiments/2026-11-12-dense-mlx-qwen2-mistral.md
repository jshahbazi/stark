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

- 2026-11-12: the branch merged `origin/main` (`0366d566`) as `7b395605`
  without conflicts. A metadata-only live re-check found every recorded
  revision still at the head. All nine repositories still pass preflight and
  header planning, with identical config and header hashes. Successor attempts
  `20261112-DENSE-MLX-02` and `-03` below carry the numerical runs; the
  canonical aggregates are in the
  [qualification run record](../archive/runs/2026-11-12-dense-mlx-qwen2-mistral-qualification.md).
  The results above remain the structural scope of attempt `-01`.

## 20261112-DENSE-MLX-02: Qwen2.5 0.5B Instruct MLX 4-bit numerical qualification

### Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-DENSE-MLX-02`; registered and executed 2026-11-12; Claude agent session on branch `claude/dense-mlx-qwen2-mistral` |
| Kind | Correctness, precision and prefix reuse |
| Attempt state | Completed |
| Result and acceptance criterion | **Failed** the preregistered gates. `dense_parity` `bf16-centered-logits-v3` passes 0/5 cases against stock MLX-LM on the same quantized weights. Worker greedy generation is exact on prompt A (13 tokens to EOS) and diverges at token 20 of 32 on prompt B. Scorer (max 0.547 against 0.75), replay, recovery and restart controls passed. Prefix-reuse exactness **passed** (32/32 requests, 6/6 invalidations) |
| Evidence scope | Full-model numerical (logits, greedy, scorer) and lifecycle (prefix reuse) on Metal; no timing |
| Promotion decision | Rejected for numerical qualification at this precision policy; awaiting a precision-matched native path |
| Disposition | Experiment only; validation tooling change adopted on branch |
| Predecessor / related attempts / governing issue / PR | `20261112-DENSE-MLX-01`; sibling `-03`; no PR |

### Hypothesis and change

- **Expected behaviour:** the BF16 Stark conversion of an MLX affine Qwen2.5
  export reproduces stock MLX-LM's logits and greedy output within the
  existing dense gates.
- **Baseline and candidate:** the reference is stock MLX-LM `1f7057ca`
  (MLX 0.32.3) on the downloaded checkpoint. The candidate is the
  `stark-convert` BF16 output on the branch's native Metal build.
- **Thresholds:** fixed before the run (private preregistration
  `6a443af7...`); unchanged afterwards.
- **Diagnostics:** an MLX reference with every parameter cast to BF16, an MLX
  F16-versus-BF16 comparison, and an F16 Stark conversion. These diagnostics
  do not change the verdict.

### Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, executed by stock MLX-LM | Same source converted with `--metadata-profile hf-generic --dtype bfloat16` |
| Format/quantization and config/weight/tokenizer/processor/template hashes | MLX affine 4-bit, group 64, tied head; all floating tensors F16; weights `ddffab9c...`, config `b045e57e...` | Output `bd822bbb...`, config `bb3b6784...`, tokenizer `9c5ae00e...`; full hashes in the run record |
| Backend, vendor revision and local patch provenance | MLX-LM `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`, MLX 0.32.3, Transformers 5.7.0; recreated lock `a98ad41b...` | Native Metal dense decoder, no Rust change |
| Stark commit, dirty patch/runtime manifest and executable hash | Not applicable | `7b395605` plus the uncommitted validation-script change committed with this record; `stark-generate` `6eea83b4...`, `stark-score` `467a33c0...`, `dense_parity` `7fcdcb2a...` |
| Toolchains, dependency lock hashes and build flags | Python 3.13.13 | Rust 1.98.1 release, locked; converter lock `df771172...` |
| Weight, activation, accumulation, recurrent-state and KV precision | 4-bit codes with F16 scales and biases; F16 activations and KV | 4-bit codes with BF16-rounded scales and biases; BF16 activations and KV; no recurrent state |
| Selected kernel and materialization policies | Stock MLX quantized matmul | Not instrumented |
| Hardware, physical RAM, OS/driver/Metal toolchain | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332) | Same |
| Available headroom, observed peak memory and measurement method | Not measured | Not measured |

### Workload and controls

- **Image capability:** N/A. The source is a text-only `Qwen2ForCausalLM`
  with no `vision_config`, and planning accepted its complete inventory.
- **Image-loaded cache and performance arms:** N/A.
- **MTP:** absent. There are no draft-head tensors, and the readiness reply
  reports `mtp.supported` false.
- **Image and MTP transitions:** N/A. The prefix harness's synthetic image
  operation invalidated the cache as documented.
- **Logit cases:** five, of 1, 39, 43, 129 and 1,029 tokens; context
  max(1,024, input + 32).
- **Generation:** two 32-token greedy generations; MTP off.
- **Prefix harness:** context 8,192, first turn about 3,000 tokens, one tool,
  `--max-new 64`, seed 11, exactness mode only.
- **Run order:** serial GPU work. Another agent's workload may have shared the
  GPU, so no timing is used.

### Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Validation tooling | Reference refused full-attention configs | Added a 1,029-token full-attention long case with tests | Tooling change; recorded |
| Gate run | Stock MLX F16 reference versus Stark BF16 | 0/5 logit cases; prompt B diverges at token 20 | Valid failure |
| BF16-cast reference | MLX with BF16 parameters versus Stark BF16 | 0/5; centered rel. L2 3.00-7.15%; prompt B diverges at token 19 | Diagnostic |
| MLX self-comparison | MLX BF16-cast versus MLX F16 | 0/5; 3.42-6.41%; KL up to 0.0129; prompt B diverges at token 19 | Diagnostic calibration |
| F16 candidate | `--dtype float16` conversion | Converts; native load `model_load_failed` | Runtime-blocked |
| Prefix reuse | `--prefix-cache` versus without, same sealed worker | 32/32 identical and rule-matching; 6/6 invalidations | Valid pass |
| Determinism | Source-profile reference rerun with final script | All reference hashes and generations identical | Binding control |

### Correctness and lifecycle

- **Logit agreement:** centered relative L2 2.86-4.84% (budget 2%). KL at most
  0.0076 and next-token argmax agrees in every case. Stock MLX rounding its
  own parameters to BF16 deviates more. This failure is attributed to running
  BF16 on an all-F16 source; no Stark defect was found.
- **Prefix reuse:** rewind reuse is exact on this artifact (2,048 to 3,072
  reused tokens), including restart, reset, error, cancellation, scope,
  context and image invalidation.
- **Tests:** `scripts/test_dense_native_validation.py` (3) and the converter
  suite; native Metal unit tests not run (no Rust change).

### Performance and memory

| Metric and measurement boundary | Baseline | Candidate | Repeats/dispersion | Eligibility and conclusion |
| --- | --- | --- | --- | --- |
| Tokenization/rendering/planning | Not measured | Not measured | None | Not measured |
| First token / prefill | Not measured | Not measured | None | Another workload may have shared the GPU |
| Useful decode / end-to-end rate | Not measured | Not measured | None | Not measured |
| Observed peak memory / retained payload | Not measured | Not measured | None | Not measured |

### Evidence, decision and limits

- **Evidence:** aggregates and receipt hashes are in the
  [run record](../archive/runs/2026-11-12-dense-mlx-qwen2-mistral-qualification.md).
- **Decision:** not qualified; do not claim numerical parity for MLX Qwen2
  artifacts.
- **Untested:** 8-bit exports, larger Qwen2.5/QwQ exports, the catalogue
  install route and tool behaviour.
- **Next action:** add native F16 (or F32) activation support for MLX
  F16-parameter sources, or adopt a reviewed precision-specific gate
  calibrated from the MLX F16/BF16 envelope, then rerun under a new attempt ID.

## 20261112-DENSE-MLX-03: Mistral 7B Instruct v0.3 MLX 4-bit numerical qualification

### Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-DENSE-MLX-03`; registered and executed 2026-11-12; Claude agent session on branch `claude/dense-mlx-qwen2-mistral` |
| Kind | Compatibility, correctness, precision and prefix reuse |
| Attempt state | Completed |
| Result and acceptance criterion | Published artifact: **failed** at conversion (`UnsupportedMetadata`: named chat-template list). Diagnostic template-substituted copy with identical weights, **failed** the logit gate 4/5 (1,029-token case: RMSE 0.107, centered rel. L2 4.02%). Worker greedy and scorer checks **passed**. Prefix-reuse rows were exact, but the aggregate failed on the template-driven follow-up rule |
| Evidence scope | Conversion; full-model numerical and lifecycle evidence on a diagnostic copy only; no timing |
| Promotion decision | Rejected; the published revision is not importable |
| Disposition | Experiment only |
| Predecessor / related attempts / governing issue / PR | `20261112-DENSE-MLX-01`; sibling `-02`; no PR |

### Hypothesis and change

- **Expected behaviour:** the published MLX export converts and matches stock
  MLX-LM within the dense gates.
- **Baseline and candidate:** the same as `-02`, applied to
  `mlx-community/Mistral-7B-Instruct-v0.3-4bit`.
- **Thresholds:** the same preregistration as `-02`.
- **Diagnostic copies:** they replace only `tokenizer_config.json`'s named
  `chat_template` list with its `default` (numerics) or `tool_use` (prefix
  harness) string. Weights are byte-identical, and these copies are not
  qualification artifacts.

### Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` @ `a4b8f870474b0eb527f466a03fbc187830d271f5` | Published: no artifact. Diagnostic: same weights, substituted template |
| Format/quantization and config/weight/tokenizer/processor/template hashes | MLX affine 4-bit, group 64, untied; all floating tensors F16; weights `d7182d88...`; `tokenizer_config.json` `e8fbcfef...` | Diagnostic output `4283ec25...`, config `a8ced577...`, tokenizer `3e017710...` |
| Backend, vendor revision and local patch provenance | As in `-02` | As in `-02` |
| Stark commit, dirty patch/runtime manifest and executable hash | Not applicable | As in `-02` |
| Toolchains, dependency lock hashes and build flags | As in `-02` | As in `-02` |
| Weight, activation, accumulation, recurrent-state and KV precision | F16 scales, biases, norms and activations | BF16-rounded scales and biases; BF16 activations and KV |
| Selected kernel and materialization policies | Stock MLX | Not instrumented |
| Hardware, physical RAM, OS/driver/Metal toolchain | As in `-02` | As in `-02` |
| Available headroom, observed peak memory and measurement method | Not measured | Not measured |

### Workload and controls

- **Image capability:** N/A. The source is a text-only `MistralForCausalLM`
  with no `vision_config`.
- **MTP:** absent. The complete inventory has no draft-head tensors.
- **Image arms:** N/A.
- **Workload:** the same cases and budgets as `-02`, with prompt lengths of 16
  and 20 tokens. The prefix harness used the `tool_use` copy, because the
  `default` template cannot render its tool history.

### Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Published conversion | `stark-convert` on the pinned download | `conversion_failed`; cause `UnsupportedMetadata` | Valid failure |
| Gate run (diagnostic copy) | Stock MLX F16 versus Stark BF16 | 4/5 logit cases; 1,029-token case fails; workers pass | Diagnostic weights evidence; gate failed |
| BF16-cast reference | MLX BF16 versus Stark BF16 | 4/5; 1,029-token case passes at 1.11%; 20-token case fails at 2.80% | Diagnostic |
| MLX self-comparison | MLX BF16-cast versus MLX F16 | 3/5; 1,029-token case: RMSE 0.108, 4.18% | Diagnostic calibration |
| Prefix (`default` template) | Exactness harness | `ChatError`; not run | Invalid setup |
| Prefix (`tool_use` copy) | Exactness harness | 32/32 identical and rule-matching; `follow_up_3` reused 0 | Rows pass; aggregate fails |

### Correctness and lifecycle

- **Diagnostic weights:** match MLX within the gate except the 1,029-token
  case. Stock MLX's own BF16 rounding deviates the same amount there, and
  greedy generation is exact on both prompts.
- **Prefix reuse:** rewind reuse is exact for every request, and all
  invalidations reuse nothing.
- **Template limit:** the template layout limits follow-up reuse. This is an
  interpretation, not a reproduced engine fault.

### Performance and memory

Not measured, for the same reason as `-02`.

### Evidence, decision and limits

- **Evidence:** the
  [run record](../archive/runs/2026-11-12-dense-mlx-qwen2-mistral-qualification.md).
- **Decision:** not qualified, and the published revision is not importable.
  By code reading, the catalogue still offers it for download.
- **Next action:** decide whether named chat templates fail closed in the
  catalogue, or are supported through the tool-template review. Then rerun on
  the published artifact under a new ID, with the precision follow-up from
  `-02`.

## 20261112-DENSE-MLX-04: Qwen2.5 0.5B Instruct MLX 4-bit, native F16 profile

The canonical aggregates for `-04`, `-05` and `-06` are in the
[F16 run record](../archive/runs/2026-11-12-dense-f16-native-qualification.md).

### Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-DENSE-MLX-04`; registered and executed 2026-11-12 US local time (2026-11-13 00:17-00:45 UTC); Claude agent session on branch `claude/engine-f16-weights` |
| Kind | Precision (new native F16 activation profile), correctness and prefix reuse |
| Attempt state | Completed |
| Result and acceptance criterion | **Passed** all preregistered gates (unchanged from `-02`). `dense_parity` 5/5 cases, centered relative L2 0.37-0.61% against the 2% budget. Both 32-token greedy generations exact. Scorer max error 0.031. Prefix exactness 32/32 with invalidations as documented |
| Evidence scope | Full-model numerical (logits, greedy, scorer) and lifecycle (prefix reuse) on Metal; no timing |
| Promotion decision | Promoted within named scope: this exact F16 artifact (`e227c073...`) on the candidate build. Not a catalogue default; the app still converts to BF16 |
| Disposition | Engine F16 profile adopted on branch `claude/engine-f16-weights`; not pushed, merged, released or installed |
| Predecessor / related attempts / governing issue / PR | Successor to `-02` (BF16, failed); siblings `-05`, `-06`; no PR |

### Hypothesis and change

- **Problem:** `-02` showed that a BF16 conversion of this all-F16 source misses
  the logit gate, and that the F16 conversion fails native load
  (`model_load_failed`).
- **Cause:** the engine fixed the dense activation dtype to BF16
  (`language_model_from_config`), so `ParameterLeaf::validate` rejected the
  first F16 tensor, the tied embedding's scale plane. The dense-path Metal
  kernels and Rust routes also had no F16 variants.
- **Change:** add an F16 activation profile. The precision policy is in the run
  record; in short, F16 weights, scales, biases, activations, KV cache and
  readout, with F32 accumulation. The profile is selected only for a decoder
  with F16 and no BF16 storage, and admitted only for the plain dense graph.
  BF16 artifacts keep their established policy.
- **Thresholds:** fixed before the run in private preregistration `2a947504...`,
  identical to `6a443af7...`.
- **Reference:** stock MLX-LM on the exact downloaded checkpoint (source
  profile, F16), the same fixture as `-02`; a rerun reproduced every reference
  logit file byte-for-byte.

### Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, BF16 conversion (`-02`) | Same source, `--dtype float16` conversion |
| Format/quantization and config/weight/tokenizer/processor/template hashes | Output `bd822bbb...` | MLX affine 4-bit, group 64, tied head; output `e227c07354092ee96211a0068e279f81ee26cc553c6f73265247cc904967be04`, config `bb3b6784...`, tokenizer `9c5ae00e...` |
| Backend, vendor revision and local patch provenance | Native Metal dense decoder at `e1e90517` | Same plus the F16 profile (this branch) |
| Stark commit, dirty patch/runtime manifest and executable hash | `e1e90517`; `stark-generate` `ec741a1e...` | Gate runs: pre-commit build of the branch change, `stark-generate` `2fa9f4df...`, `stark-score` `c6205b40...`, `dense_parity` `26f85d0e...`. Engine commit `f203f004` rebuilt (`2adf7b0d...`, `b73e1fea...`, `224d9571...`) reproduces every logit hash, worker output and prefix row |
| Toolchains, dependency lock hashes and build flags | Rust 1.98.1 release, locked | Same; `Cargo.lock` `f836a99b...` |
| Weight, activation, accumulation, recurrent-state and KV precision | BF16-rounded scales and biases; BF16 activations and KV | F16 scales, biases, norms and embeddings; F16 activations, KV cache and readout; F32 accumulation; no recurrent state |
| Selected kernel and materialization policies | Not instrumented | Generic F16 GEMV/GEMM (F32-accumulating), F16 attention kernels; BF16-only source, A8 and MXU-attention paths unused |
| Hardware, physical RAM, OS/driver/Metal toolchain | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332) | Same; Metal 32023.883 |
| Available headroom, observed peak memory and measurement method | Not measured | Not measured |

### Workload and controls

- **Image capability, image arms and MTP:** N/A, as in `-02`. The source is a
  text-only `Qwen2ForCausalLM` with no draft-head tensors.
- **Workload:** the same five logit cases, two 32-token greedy generations,
  scorer cases and prefix-harness flags as `-02`.
- **Run order:** serial. Another agent's Phi workload may have shared the GPU,
  so no timing is used.

### Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Load reproduction | Earlier F16 artifact on the baseline build | `Invalid tensor ... @ F16, expected ... @ BF16` | Root cause |
| Functional smoke | Candidate load and finite logits for synthetic IDs | Loaded, finite | Pre-registration smoke; no comparison |
| Gate run | Stock MLX F16 reference versus Stark F16 | All three gates pass | Valid pass |

### Correctness and lifecycle

- **Logits:** centered relative L2 0.37-0.61%, KL at most 0.00014, and argmax
  agrees in every case.
- **Prefix reuse:** rewind reuse is exact on the F16 KV cache: follow-ups reuse
  2,048 tokens and the maximum is 3,072.
- **Tests:** the new engine and converter tests are listed in the run record.

### Performance and memory

Not measured (another workload may have shared the GPU).

### Evidence, decision and limits

- **Decision:** the F16 artifact of this export meets the dense gates on this
  build. Do not transfer this result to BF16 conversions, 8-bit or larger
  Qwen2.5/QwQ exports, or later builds.
- **Untested:** catalogue and app routes, tool calling, timing, sampled
  generation.

## 20261112-DENSE-MLX-05: Mistral 7B Instruct v0.3 MLX 4-bit diagnostic copy, native F16 profile

### Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-DENSE-MLX-05`; same session and dates as `-04` |
| Kind | Precision, correctness and prefix reuse |
| Attempt state | Completed |
| Result and acceptance criterion | Diagnostic copy: logit gate **passed** 5/5 (0.08-0.34%), worker checks **passed** (both greedy-exact, scorer 0.023/0.031), prefix rows 32/32 exact but aggregate **failed** on the template follow-up rule, as in `-03`. Published revision: still **fails conversion** |
| Evidence scope | Full-model numerical and lifecycle evidence on a diagnostic copy with byte-identical weights; no timing |
| Promotion decision | Rejected for the published revision (not importable); diagnostic only for the copy |
| Disposition | Experiment only |
| Predecessor / related attempts / governing issue / PR | Successor to `-03`; siblings `-04`, `-06`; no PR |

### Hypothesis and change

The hypothesis and change are as in `-04`, on the `-03` diagnostic copy, which
substitutes the source's own `default` template. The prefix harness uses the
`tool_use` copy, as `-03` did. The thresholds and reference fixture are the
same as `-03`; a reference rerun reproduced every logit file byte-for-byte.

### Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` @ `a4b8f870474b0eb527f466a03fbc187830d271f5`, diagnostic copy, BF16 (`-03`) | Same copy, `--dtype float16` |
| Format/quantization and config/weight/tokenizer/processor/template hashes | Output `4283ec25...` | Output `968cf950b4d7cba8ab2f0d1c97a708f0a1bfe2dc9f8f209aa6d8492e671d1a5a`, config `a8ced577...`, tokenizer `3e017710...`; `tool_use` copy config `ca215e8a...`, same output hash |
| Remaining rows | As in `-04` | As in `-04` |

### Workload and controls

The workload and controls are as in `-03`, with prompt lengths of 16 and 20
tokens. Image capability and MTP are N/A.

### Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Gate run (diagnostic copy) | Stock MLX F16 versus Stark F16 | 5/5 logit cases; workers pass | Diagnostic weights evidence |
| Prefix (`tool_use` F16 copy) | Exactness harness | 32/32 identical and rule-matching; follow-ups reuse 0, 2,048, 2,048, 0 | Rows pass; aggregate fails as in `-03` |

### Performance and memory

Not measured.

### Evidence, decision and limits

- **Decision:** the F16 profile removes the 1,029-token logit failure of `-03`
  on these weights. The published export remains blocked by its named template
  list, so no artifact is qualified.
- **Next action:** the named-template decision from `-03`.

## 20261112-DENSE-MLX-06: BF16 regression control for the F16 engine change

### Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-DENSE-MLX-06`; same session and dates as `-04` |
| Kind | Correctness (regression control) |
| Attempt state | Completed |
| Result and acceptance criterion | **Passed**: every BF16 logit hash and worker output hash is bit-identical between the baseline (`e1e90517`) and candidate builds, and identical to the `-02`/`-03` receipts |
| Evidence scope | Full-model numerical and worker outputs on existing BF16 artifacts; no timing |
| Promotion decision | Supports adopting the engine change without altering any BF16 path |
| Disposition | Adopted on branch |
| Predecessor / related attempts / governing issue / PR | `-02`, `-03`; no PR |

### Workload and results

| Artifact | Comparison | Result |
| --- | --- | --- |
| BF16 Qwen2.5 MLX `bd822bbb...` | `dense_parity` hashes (5 cases), worker generations and scorer, baseline versus candidate and the `-02` receipt | Identical |
| BF16 Mistral diagnostic copy `4283ec25...` | `dense_parity` hashes (5 cases), worker generations and scorer versus the `-03` receipt | Identical |
| `Qwen/Qwen3-0.6B` @ `c1899de2...` BF16, `87c3344a...` | Next-token logit hashes for three synthetic inputs; `stark-generate` readiness and two 32-token replies | Identical |
| Qwen3-0.6B F16 conversion `92fa34d8...` (query/key norms) | Admission negative control | Candidate refuses with `UnsupportedPrecision(F16)`; baseline refused with the dtype mismatch |

### Limits

- No other BF16 dense artifact was available locally without a download.
- Kernel-level and full-suite engine tests are in the run record.
