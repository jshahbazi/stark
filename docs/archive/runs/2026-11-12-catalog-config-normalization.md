# Catalogue config normalization

Date is 2026-11-12. Implementation starts from `32d1582b` on
`claude/model-catalog-search-ux-9f83b3`, on branch
`claude/catalog-config-normalizer`. Ledger entry:
[20261112-CATALOG-01](../../model-experiments/2026-11-12-catalog-config-normalization.md).

## Cause

A crawl of the first two **Model Catalog** pages of every family found 97 rows
marked **Unavailable** only because a family planner rejected `config.json`
contents. Most differences were publisher annotations, sampling defaults or a
newer Transformers serialization of an already admitted setting.

## Implemented behavior

- `stark_convert/config_normalization.py` provides one entry point,
  `normalize_config(family, config)`, returning a new configuration and fixed
  change labels. The caller's dictionary is never modified.
- The Qwen3, dense (Qwen2, Llama, Mistral), Gemma 2/3 text, Gemma 4, OLMo 3,
  Phi-3, Granite/SmolLM3, Qwen3.5 and LFM2 planners call it on the single line
  where they read their configuration. Header-only catalogue admission and
  source-backed conversion therefore judge the same normalized configuration.
  No other planner line changed.
- The reviewed rules and what still fails closed are listed in
  [Model support](../../MODEL_SUPPORT.md). The attention dropout rewrite is
  isolated behind `ZERO_INACTIVE_ATTENTION_DROPOUT`.

## Live observations

Live Hugging Face metadata, unauthenticated, one pass over the 97 rows with at
least 1.5 seconds per repository. Only `config.json`, the shard index and
tensor headers were read; no weights were downloaded. Each row was planned with
`planning.plan_from_headers` in text mode against its current head revision.

| Family | Rows | Now plan |
| --- | --- | --- |
| SmolLM3 | 34 | 25 |
| Granite | 15 | 7 |
| LFM2 | 15 | 12 |
| Gemma 2 | 10 | 9 |
| OLMo 3 | 5 | 2 |
| Llama | 4 | 2 |
| Qwen3.5 | 4 | 4 |
| Gemma 4 text | 4 | 0 |
| Gemma 3 text | 3 | 0 |
| Mistral | 2 | 1 |
| Phi-3 | 1 | 1 |
| Total | 97 | 63 |

All 97 rows loaded. The per-row outcome matched the prototype simulation
exactly, including the 12 LFM2 rows that depend on the two LFM2 rules.

After the live pass, the full suite showed that the Qwen3 planner deliberately
refuses a nested theta without `rope_type`. Flattening was then withdrawn for
the three planners that validate the plain nested form themselves (Qwen3,
Llama and LFM2). No live row was re-run. Of the passing rows, only the LFM2
ones had a flattened `rope_parameters` among those families, and each
declares `rope_theta` with a default `rope_type`, which the LFM2 planner
accepts unchanged; that outcome is inferred from the planner source, not
re-measured.

Remaining rejections, by planner message:

| Rows | Family | Message |
| --- | --- | --- |
| 8 | Granite | Granite requires tied word embeddings |
| 3 | Gemma 4 text | Gemma 4 text config has unimplemented options |
| 3 | OLMo 3 | OLMo 3 nested RoPE parameters are unsupported |
| 3 | SmolLM3 | RoPE scaling is outside qualified dense coverage (two differing declared RoPE bases) |
| 2 | SmolLM3 | SmolLM3 sliding-window layer boundary is invalid |
| 2 | SmolLM3 | SmolLM3 requires an integer per-layer no_rope_layers mask |
| 2 | Gemma 3 text | New Gemma RoPE parameters require a separate adapter |
| 2 | LFM2 | LFM2 config options are outside implemented coverage |
| 1 | LFM2 | LFM2 checkpoint is missing a required tensor |
| 1 | Llama | Dense checkpoint has unexpected tensors |
| 1 | Llama | RoPE scaling type is outside qualified dense coverage |
| 1 | Mistral | Layer attention types disagree with qualified sliding-window semantics |
| 1 | Gemma 2 | Gemma config requires boolean use_bidirectional_attention |
| 1 | Gemma 3 text | Gemma checkpoint is missing a required tensor |
| 1 | Gemma 4 text | Gemma 4 text architecture is outside coverage |
| 1 | SmolLM3 | Dense checkpoint is missing a required tensor |
| 1 | SmolLM3 | SmolLM3 config has malformed layer_types |

## Validation

- New `test_config_normalization.py`: 67 model-free tests covering each rule,
  its refusal cases, no input mutation, idempotence, out-of-scope families, and
  for 13 families a published-form configuration that the planner rejects
  without the pass, plans identically to the clean fixture through both the
  source-backed and header-only paths, and is still refused with an unknown key.
- Four existing fail-closed tests that expect a planner to refuse
  `attention_dropout: 0.1` (dense, Gemma, Phi, Qwen3.5) now switch
  `ZERO_INACTIVE_ATTENTION_DROPOUT` off, so the planners' own check stays
  covered independently of the normalizer's rewrite.
- Full converter suite without the native `stark-template` build: no failure
  or error outside the baseline set recorded for `32d1582b` in this checkout.
- Converter Ruff lint and format checks and the repository policy check passed.
- No row was converted, loaded or executed. Native loading, numerical
  correctness, cache, image and MTP qualification of newly admitted artifacts
  remain separate and unverified. Image-mode admission was not exercised live.
