# Beginner published task benchmark references

Date: 2026-11-08. Implementation source:
`9c909960ad6f0b20573a8bb4b6990a0f1a0e34db` on
`codex/beginner-hardware-ranking`.

## Scope and behavior

Beginner Balanced now uses a packaged, offline snapshot of published upstream
LiveCodeBench, BFCL and IFEval aggregate results. This run imported public
reference scores; it did not execute any benchmark or establish quality of a
Stark artifact. The implementation changes recommendation metadata and ordering,
plus the Swift details display. It does not change inference, numerical policy,
prefix reuse or model admission.

Resource admission and the existing deterministic capacity/memory fallback
remain first. A single task cohort can reorder only its scored slots within a
memory-headroom tier; unknown choices retain their original positions. Cohort
selection uses eligible parent-model coverage, publisher authority and ID,
without examining score magnitude. Duplicate exports of one parent do not
create comparison coverage. Explicit Lighter model and More model capacity
preferences keep their resource objectives.

The report names the narrow signal, retains evaluated-parent and candidate
identities, includes source/settings and labels public results as upstream
references. A model-repository match does not establish evaluated-revision or
artifact equivalence. Quantized parent references require an explicit reviewed
repository/revision binding. The Swift reader preserves older reports and
bounds the optional details to 16 strings of at most 2,000 UTF-8 bytes each.

## Public source snapshot

The registry contains 54 aggregate rows across five separate cohorts.
`recommendation_benchmarks.json` SHA-256:
`b72cd716b75957317b668d9be0029b1b21fc8a520f98169bc3027e265662d799`.

| Source | Coverage and provenance |
| --- | --- |
| [BFCL v4 overall CSV](https://gorilla.cs.berkeley.edu/data_overall.csv) | Official 2026-04-12 snapshot; 33 FC, 11 Prompt and one FC-thinking row. These remain separate cohorts. Evaluator `f7cf735`, reproducibility package `bfcl-eval==2025.12.17`. CSV SHA-256 `ede5081a24bc5ac7f9b3b09a52af6bcfcd35aa0f1709239fed979db77a8159a6`. |
| [HF Open LLM Leaderboard contents](https://huggingface.co/datasets/open-llm-leaderboard/contents/tree/9c09a7cae43334062a82cb164f2ef255013dafa2) | Seven Qwen2.5-Instruct BF16 chat rows, unflagged, with evaluated model revisions. `/rows` and `/info` both returned `x-revision: 9c09a7cae43334062a82cb164f2ef255013dafa2`; the dataset API confirmed the same SHA. The registry records the immutable Parquet URL/hash and extraction response hashes. Its metric remains the dataset's published IFEval score; strict prompt vs instruction aggregation was not identified. |
| [Qwen3.8 official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md) | LiveCodeBench v6 developer comparison table: Flash-Next 91.9 and 27B 90.3. Source README SHA-256 `5c151489c5de6ba51318e5e03d9ffb80622f118b37e7af550a6c174c5a9fe104`. The independent 27B card's pinned URL/hash is also retained. Window, pass@k, sampling, execution policy and aggregation are undisclosed. This cohort is not mixed with official leaderboard windows or other publishers. |

Four reviewed, pinned export cards bind the Jundot Qwen3.8-27B export and the
pipenetwork, Lily and MLX-Serve native Qwen3.8-Flash-Next exports to their
explicit parents. Their README URLs and hashes are retained in the registry.
These bindings transfer reference attribution, without transferring measured
quality or native validation.

Coverage remains incomplete. Missing scores are unknown. IFEval is an
instruction-following signal, BFCL measures tool use under its own harness,
and coding-problem results do not establish repository-task completion. The
[current recommendation policy](../../MODEL_RECOMMENDATIONS.md) describes the
comparison and future artifact evidence requirements. MaxxBench remains
deferred until model-specific user runs exist; no MaxxBench runner or scores
were added.

## Validation

All checks are model-free. This work fetched public score and metadata sources;
no weights were downloaded and no model was loaded. It ran no inference or
task evaluations and did not replace the installed app.

| Check | Result |
| --- | --- |
| Focused benchmark registry and guided model tests | 100 passed |
| Converter suite on the final implementation commit | 6,734 passed; 96 skipped; 265.31 seconds |
| Native Beginner UI/session/report harness | 11 passed |
| Menu-bar controller harness | 134 passed |
| Menu-bar builder suite | 46 passed |
| All menu-bar Swift sources, macOS 26.4 arm64 typecheck | Passed |
| Converter Ruff lint and format | Passed; 181 files formatted |
| Repository policy and diff whitespace checks | Passed |

Regression checks cover hardware exclusion before scores, score/version
separation, deterministic ties, unknown-slot preservation, distinct parent
coverage, exact export bindings, invalid snapshot fallback, score bounds,
optional Swift metadata bounds and explicit resource preferences. An integration
check using imported BFCL results promotes Qwen3-8B (42.57) above Qwen3-14B
(41.03) for tool use, while the capacity fallback prefers 14B. Its resource plans
are synthetic fixtures and do not claim measured model fit.
