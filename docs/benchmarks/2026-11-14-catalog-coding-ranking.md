# Catalog coding ranking snapshot

The Top Models tab displays the ten family entries approved after the
November 14, 2026 shared-benchmark experiment. Its static data is owned by
`TopModelRanking` in `apps/menu-bar/Sources/StarkApp.swift`. It does not change
Model Catalog sorting or perform network requests to update rankings.

## Source and attribution

The input is [llmbench](https://llmbench.site/), generated at
`2026-11-14T18:14:50Z`. The [data downloads](https://llmbench.site/data/) and
[method](https://llmbench.site/method/) describe the compilation. llmbench's
compilation is CC BY-SA 4.0, incorporating whatmodel.sh by Merlinsbeard Pty Ltd;
underlying sources retain their own licenses. Derived ranking data retains
that attribution and compilation license. The aggregate JSON records each
input source's attribution and license, including non-commercial source terms;
this calculation does not grant additional rights to underlying data.

Inputs were the public `data/index.json` and `data/cells/<source>.json` files
for whatmodel, models-dev, vals, aider-polyglot, arena, epoch, livebench,
swe-bench and terminal-bench. Their SHA-256 hashes, the local conversion-roster
hash, model-level fitted indexes and family aggregates are retained in
[the aggregate record](data/catalog-coding-ranking-2026-11-14.json).
These hashes bind this calculation to its input snapshot; re-fetching the live
URLs may produce different inputs.

## Calculation

1. Match the locally registered reference repositories to llmbench model
   repository URLs, using Stark configuration classes for family membership.
   Add the pinned Flash Next and GLM native entries. Resolve Devstral Small
   2505 by exact model ID. It has no qualifying connected comparison here.
   Exclude quantization copies, duplicate registrations and the unsupported
   A3B candidates. This is a scored subset of the local roster, not a snapshot
   of every dynamically discoverable Hugging Face model.
2. Select headline coding outcomes with increasing percentage/fraction scores.
   Exclude quantized/search-flagged runs, Elo, cost, latency, explicit
   pass@5/pass@10 columns and unspecified/unversioned benchmark windows.
   Use llmbench's best eligible run convention. Keep one source column for
   each linked benchmark identity, preferring the first column in the index
   (usually whatmodel). This avoids duplicate imports of the same benchmark.
3. For each model pair, compare only identical source-and-version columns
   present for both models. Convert fractions to percentages. Within each
   benchmark group, average A's score minus B's score; average those group
   margins equally. HumanEval variants, MBPP variants, LiveCodeBench versions,
   SWE-bench variants, Terminal-Bench versions, code-migration versions and
   vibe-code versions each share a group; other declared metric families
   share their metadata group. Distinct columns are never substituted for one
   another when constructing a direct comparison.
4. Require two shared benchmark groups for an edge. Its weight is the number
   of shared groups, capped at five. Fit model indexes by minimizing
   `sum(weight * (index[A] - index[B] - shared_margin[A,B])^2)`.
   Anchor Flash Next at 100 and retain its connected component. The retained
   graph has 28 models and 167 edges. Indirect paths connect models with no
   direct overlap. Each percentage-point margin has equal scale; this is an
   additive approximation across tasks with different difficulty.
5. Average connected model indexes equally within each family. Quantizations
   do not receive extra weight. Display one decimal place, preserving the
   approved descending order. No missing score becomes zero.

## Approved entries

| Family ID | Family | Index | Scored models |
| --- | --- | ---: | ---: |
| qwen4_exp | Qwen3.8 Flash Next | 100.0 | 1 |
| glm5_next | GLM-5.3 Flash | 97.3 | 1 |
| qwen3_5 | Qwen3.5 dense | 61.1 | 6 |
| qwen3 | Qwen3 dense | 48.3 | 3 |
| qwen2 | Qwen2 dense | 41.0 | 6 |
| phi3 | Phi-3 dense | 34.9 | 1 |
| gemma3_text | Gemma 3 text | 25.2 | 3 |
| mistral | Mistral dense | 18.3 | 3 |
| llama | Llama dense | 13.3 | 3 |
| gemma2 | Gemma 2 dense | 1.3 | 1 |

## Interpretation

The index is an offset ranking scale anchored at Flash Next = 100. It is not
a pass percentage, probability, measured Stark coding score or model-specific
download recommendation. Indexes outside 0–100 are possible in this method.
No hardware, memory-fit, inference-speed or quantization adjustment is applied.

Family membership follows the app's architecture grouping: Qwen3.5 includes
the scored dense Qwen3.6/3.8 27B models; Phi-3's scored entry is Phi-4 Mini.
Scored sizes and tuning types differ across families, and missing or
single-benchmark models are excluded. Granite, SmolLM3, LFM2 and LLaMBA have
no connected qualifying index in this calculation; ModernBERT is a classifier.

Matching a benchmark name/source/version does not establish identical agents,
reasoning effort, context budget or evaluation conditions. The best-run rule
can favor models with more reported runs. Flash Next and GLM's direct overlap
has two groups whose margins point in opposite directions; their fitted
indexes also reflect indirect paths and should be treated as close.

The aggregate record includes leave-one-group-out rank ranges. Trials that
disconnect a family's scored models are counted separately; ranges omit those
trials. The strict evidence threshold excludes more small Qwen3 models, which
raises its family average relative to the one-group exploratory result.
The snapshot preserves the approved result without claiming statistical
confidence intervals or a universal model-family ordering.
