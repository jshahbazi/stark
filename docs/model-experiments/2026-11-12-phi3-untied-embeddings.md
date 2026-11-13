# 20261112-PHI-01: untied Phi3 output head

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261112-PHI-01` / 2026-11-12 / 2026-11-12 / Claude agent on branch `claude/phi-untied-embeddings` |
| Kind | Compatibility and correctness (new embedding layout for an existing family) |
| Attempt state | Completed |
| Result and acceptance criterion | **Failed** for `microsoft/Phi-3.5-mini-instruct`: the native worker and scorer cannot execute the artifact because its attention head dimension (96) has no Metal attention kernel. The untied conversion itself completed. **Passed** for a synthetic untied control with head dimension 128 (full-vocabulary logits, greedy/lifecycle worker checks and prefix-reuse exactness). Criteria are listed below. |
| Evidence scope | Structural (converter, header planning, tied byte identity), synthetic full-model numerical and lifecycle controls, and an actual-artifact native load/execution attempt |
| Promotion decision | Converter untied path adopted on branch; Phi-3.5-mini rejected (refused in preflight); every real untied Phi3 checkpoint awaiting qualification |
| Disposition | Adopted on branch; not merged, released or installed |
| Predecessor / related attempts / governing issue / PR | [20261112-CATALOG-01](2026-11-12-catalog-config-normalization.md) and [20261110-CATALOG-01](2026-11-10-catalog-family-admission.md) (catalogue admission); no PR opened |

## Hypothesis and change

- Concrete problem, trigger and expected behavior: 39 Phi-3-family catalogue rows were unavailable with "This family currently implements tied embeddings only", including `microsoft/phi-4`, `microsoft/Phi-4-reasoning-plus` and `microsoft/Phi-3.5-mini-instruct`. Untied `phi3` checkpoints should plan a separate output projection and run on the native engine.
- Baseline and candidate difference: the baseline converter (`origin/main` `0366d566`) rejected untied `phi3` in preflight (`unsupported_embedding_layout`) and in the planner. The candidate plans `decoder.embedding.input_embedding` (layout `input_output`) and `decoder.embedding.output_embedding` (layout `output_input`) with `UntiedEmbeddingConfig`: the same tensor names, layouts and operation order as the dense planner's untied path. The engine already loads that layout through its generic untied readout, and no engine source changed. The candidate also defaults an absent `partial_rotary_factor` to 1.0, matching Transformers 5.20.1 `Phi3Config` and MLX-LM `phi3.ModelArgs`. It rejects a `sliding_window` shorter than `max_position_embeddings`, because the plan emits full attention. After the actual-artifact failure below, preflight also refuses `phi3` configs whose head dimension is not 64, 128, 256 or 512.
- Correctness metrics, fixed thresholds, controls and references set before the run: (1) tied plans byte-identical to the baseline adapter. (2) `scripts/dense_native_validation.py workers` against a private MLX-LM fixture: 32-token greedy outputs exactly equal on both prompts, with same-worker replay, invalid-request recovery and restart replay exact, and scorer candidate logits within 0.75 absolute of the reference, with consecutive-cache replay and invalidation checks. (3) `scripts/validate_generation_prefix_cache.py --reuse rewind --mode exactness` passing all its exactness and lifecycle checks. The full-vocabulary `dense_parity` example (acceptance `bf16-centered-logits-v3`) was added as a diagnostic after the worker failure. The synthetic controls and the swapped-head negative control were added after that failure, to separate the head-dimension cause from the untied readout.
- Distinguish qualification artifacts and implementation sources from the source used to generate a reference: implementation follows the repository's dense untied path. The reference is MLX-LM `72be8c955d9a0d3099bd01865af288e0e947c3fa` (`phi3.py`, `SuScaledRoPE`, MIT, Apple) loading the same source checkpoint through `scripts/phi3_mlx_reference.py`. It runs from the locked `tools/stock_benchmark/mlx_current` project, materialized offline from an existing local uv cache.
- Why the attempt was stopped, rejected or superseded, if applicable: Phi-3.5-mini qualification stopped at native execution. A head-dimension-96 attention kernel is new Metal kernel work outside this change.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | Same source | `microsoft/Phi-3.5-mini-instruct` at `2fe192450127e6a83f7441aef6e3ca586c338b77` (MIT); synthetic controls described below |
| Format/quantization and config/weight/tokenizer/processor/template hashes | Same source | BF16 Safetensors. Source config `224de4f6…4974`, shards `c5214cdb…8837` and `41246eed…7344`, tokenizer `9e2ae3d6…6f`, tokenizer config `92a6e3f8…44`, template `78d976a4…2a90`, no processor. Converted: config `9407d6c2…f1`, weights `12848793…c6`, tokenizer `3cb815b9…47`, `conversion.json` `f9044aa3…21` |
| Backend, vendor revision and local patch provenance | Generic converted-decoder Metal engine | Same; no engine or vendor change |
| Stark commit, dirty patch/runtime manifest and executable hash | `origin/main` `0366d566` | Converter `b9e2822e` (conversion, before the head-dimension gate), reference script `6469f481`, gate `329d4779`. Rust sources unchanged from `0366d566`. `stark-generate` `073ef628…edd`, `stark-score` `217a6b9e…a98`, `stark-template` `886769ce…2ac`, `dense_parity` example `bb860bf1…925b` |
| Toolchains, dependency lock hashes and build flags | Same | Rust 1.98.1 release, `stark-score --features optimized`; converter lock `df771172…d5`; MLX reference lock `8a773ac9…0a6ac` (MLX 0.32.4, Transformers 5.20.1, Python 3.13.13) |
| Weight, activation, accumulation, recurrent-state and KV precision | n/a | BF16 weights and embeddings; F32 normalization scales. Activations, accumulation and KV as selected by the generic loader for a BF16 artifact (not separately inspected). No recurrent state. The reference is MLX BF16 weights and activations. |
| Selected kernel and materialization policies | n/a | Generic attention dispatch: head dimension 96 matches no single-pass, two-pass, GEMM or fallback kernel. LongRoPE long factors are selected statically because 131,072 > 4,096. |
| Hardware, physical RAM, OS/driver/Metal toolchain | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332), Metal 32023.883 | Same |
| Available headroom, observed peak memory and measurement method | Not measured | Not measured; another agent's MLX workload may have been running |

## Workload and controls

- Default install/load mode and actual image capability; image N/A source evidence when applicable: N/A. `Phi3ForCausalLM`, with no vision configuration keys and no processor files in the pinned revision. The image model is the separate `microsoft/Phi-3.5-vision-instruct` repository (`phi3_v`), which this route does not cover.
- Cache exactness on the default image-loaded worker; fresh/reference outputs, reused counts and complete restoration scope: no image worker exists. Text prefix reuse is the existing rewind route for converted full-attention decoders; the untied head adds no state. Tested on the synthetic head-dimension-128 artifact only (below).
- Matched text-only versus image-loaded performance arms: N/A (text-only source).
- Source MTP head/config inventory: N/A. The source has 195 tensors (input embedding, `lm_head`, final norm and 6 per layer for 32 layers); no MTP or draft tensors and no MTP configuration.
- Text-MTP/cache → image → text recovery: N/A (no image route, no MTP).
- Tool calling: the codec template (`78d976a4…2a90`) renders only system, user and assistant text and drops tool roles. `tool_support_for` reports `unavailable`. `scripts/validate_tool_calling.py` was not run; there is no protocol to qualify.
- Task/mode and fixtures: two fixed chat prompts from the dense validation recipe, and five next-token cases of 1, 14, 17, 129 and 1,029 tokens (the last crosses one 1,024-token prefill chunk). Private fixture `cases.json` `b8999ca2…f3d3`; Phi-3.5 `generations.json` `b49525e8…577c`.
- Budgets: 32 greedy output tokens with EOS stop; generation context 1,024; scorer context `max(1024, input + 32)`. The prefix validator used context 8,192, first-turn target 2,600 (actual 4,663), 70 instruction paragraphs, 24 tools, follow-ups 40/25/80 and seed 11.
- Sampling, seeds, stop rules, MTP/draft depth: greedy for the reference comparison; validator defaults for seeded stochastic cases; MTP off (none exists).
- Run order and concurrency: serial; one worker at a time. The download competed with another network workload.
- Commands: `scripts/phi3_mlx_reference.py`, `scripts/dense_native_validation.py workers`, `target/release/examples/dense_parity` and `scripts/validate_generation_prefix_cache.py`, all at the hashes above; private receipts live under the ignored `runtime/phi35/`.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Live headers, before change | Baseline preflight and header planning for Phi-3.5-mini, Phi-3-mini-4k, phi-4 and Phi-4-reasoning-plus | All four `unsupported_embedding_layout` in preflight. The planner would then reject the first three for a missing `partial_rotary_factor` and all of phi-4, Phi-4-reasoning-plus and Phi-3-mini-4k for missing LongRoPE | Valid structural evidence |
| Converter change | Untied plan, `partial_rotary_factor` default, sliding-window guard | Phi-3.5-mini plans from live headers as `UntiedEmbeddingConfig`; the other three still stop at "Phi config requires LongRoPE scaling parameters" | Valid structural evidence |
| Tied identity | Baseline and candidate adapters on 16 fixture/config combinations, plus live `microsoft/Phi-4-mini-instruct@cfbefacb` headers | All 16 written outputs and decoder configs identical; live header plan equal (194 operations) | Passed |
| Download | 7,642,181,880 weight bytes; the Xet transfer ran at about 0.4 MB/s and was restarted over plain HTTPS | Completed; shard hashes match the pinned LFS object IDs | Valid |
| Conversion | `stark-convert` with `--metadata-profile hf-generic --architecture phi3` at `b9e2822e` | `ok`, 195 source tensors, output `12848793…c6` | Valid |
| MLX-LM reference | Five logit cases and two greedy generations | Passed; generations ended at EOS after 14 and 21 tokens | Valid reference |
| Native workers | `dense_native_validation.py workers` | `native_generation_failed`; a direct synthetic request reproduced it; tokenization works | **Failed** |
| Native scorer | `dense_parity` full-vocabulary comparison | `native_score_decoder` before any comparison | **Failed** (same cause) |
| Diagnosis | Source review of Metal attention dispatch | Single-pass and two-pass kernels are compiled for 64/128/256/512, prefill GEMM for 64/128/256 and the fallback for 512 | Concrete blocker |
| Synthetic controls | Two random-weight, 2-layer untied Phi3 checkpoints with the Phi-3.5 tokenizer and LongRoPE, differing only in head dimension (128 versus 96) | 128 generates normally; 96 fails exactly like Phi-3.5-mini | Isolates the head dimension as the cause |
| Synthetic numerical | MLX-LM reference and `dense_parity` on the 128 control | All 5 cases pass: max absolute error ≤ 0.01172, RMSE ≤ 0.00225, KL ≤ 2.53e-6, greedy agreement 5/5 | Passed (synthetic only) |
| Negative control | Same 128 control with `lm_head` bytes replaced by the input embedding, against the same reference | Fails all 5 cases: max absolute error 1.80 to 2.20, KL about 0.10, greedy disagreement 5/5 | Passing admission negative control: the comparison discriminates the head |
| Synthetic lifecycle | `dense_native_validation.py workers` on the 128 control | Greedy 32/32 exact on both prompts; replay, invalid-request recovery and restart replay exact; scorer max error 0.0078 at 14 and 1,029 tokens with cache replay, invalidation and failure clearing | Passed (synthetic only) |
| Synthetic prefix reuse | `validate_generation_prefix_cache.py --reuse rewind --mode exactness` on the 128 control | 32/32 cases successful, identical to fresh, identical to the baseline worker and matching the reuse rule; 19 reused (max 5,120); six invalidation paths as documented. The baseline and candidate are the same executable. | Passed (synthetic only) |
| Preflight gate | Refuse `phi3` head dimensions without a kernel | Phi-3.5-mini and Phi-3-mini-4k now `unsupported_native_geometry`; local conversion of the 96 control refused (`unsupported_format_or_config`) | Adopted on branch |

## Correctness and lifecycle

- Reference, numerical tolerance and tested scope: MLX-LM long-factor LongRoPE. Transformers and the official remote code use short factors up to 4,096 tokens, so neither the native path nor this reference reproduces those dynamic semantics. The divergence predates this change and also applies to tied Phi-4-mini. Aggregate results are in the history table.
- Model-free versus actual artifact execution: actual Phi-3.5-mini execution failed before any numerical comparison. The passing numerical, lifecycle and prefix results are synthetic random-weight controls. They exercise the converter, the untied readout and the generic decoder, not a trained checkpoint.
- Prefix reuse: rewind on full-attention KV rows; the untied head adds no retained state. Disposition for untied Phi3 with a supported head width: implemented, artifact qualification pending. Phi-3.5-mini: blocked by the head-dimension kernel gap, not by prefix restoration.
- Context/resource rejection, cancellation, reset/restart: covered for the synthetic control by the prefix validator and worker checks.
- Tests run: the converter suite and focused Phi, model-support, header-planning, catalogue and CLI tests; see the PR description.

## Performance and memory

| Metric and measurement boundary | Baseline | Candidate | Repeats/dispersion | Eligibility and conclusion |
| --- | --- | --- | --- | --- |
| Tokenization/rendering/planning | Not measured | Not measured | n/a | n/a |
| First token / prefill | Not measured | Not measured | n/a | Artifact did not execute |
| Useful decode / end-to-end rate | Not measured | Not measured | n/a | n/a |
| Observed peak memory / retained payload | Not measured | Not measured | n/a | n/a |

## Evidence, decision and limits

- Source-bound public aggregates and private receipt hashes: Phi-3.5 reference report `289afa7d…5e5b`, workers report `f2facaff…4cb1`, dense parity `85a65695…0bb3`. Synthetic sources `c45ebbae…a3` (128) and `4f92f8c8…08` (96). Converted 128 `77453d2a…01`. Synthetic receipts: reference `8e33cc4f…65`, parity `d1f7d587…50`, workers `41cde652…e0`, prefix `4e55fd0c…15`, negative parity `ebf9d9b1…80`.
- What worked: the untied plan and its native readout, on a synthetic head-dimension-128 model, against MLX-LM. What did not: executing Phi-3.5-mini (head dimension 96).
- Adoption/default decision: the converter untied path is adopted on branch. Phi-3.5-mini and Phi-3-mini (head dimension 96) are refused in preflight with `unsupported_native_geometry`.
- Untested: every real untied Phi3 artifact, Phi-4 and Phi-3-medium included. Also untested: contexts beyond 1,061 tokens for the logit comparison, or beyond 5,907 tokens for prefix reuse, and speed and memory.
- Remaining blockers seen in live headers: phi-4 and Phi-4-reasoning-plus have no `rope_scaling` (plain RoPE is not implemented for `phi3`). Phi-3-medium-128k-instruct declares `su`, which Transformers 5.20.1 rewrites to `longrope` but the planner refuses. Phi-3/3.5 mini need a head-dimension-96 attention kernel.
- Next action: qualify an untied Phi3 checkpoint with a head dimension of 128 once its RoPE form is admitted, or add and qualify head-dimension-96 attention kernels.

Keep prompts, completions, token vectors, probabilities, credentials, exception dumps and model/private evaluation files out of this document.

## Follow-up history

None.
