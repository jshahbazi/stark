# 20261112-CATALOG-01 — shared config normalization before family planners

Registered on 2026-11-12. The canonical validation record is the
[config normalization run](../archive/runs/2026-11-12-catalog-config-normalization.md).
Predecessor: [20261110-CATALOG-01](2026-11-10-catalog-family-admission.md).

## Scope and identity

Compatibility change to catalogue admission and source-backed conversion for
the generic converter families Qwen3, Qwen2, Llama, Mistral, Gemma 2, Gemma 3
text, Gemma 4 text, OLMo 3, Phi-3, Granite, SmolLM3, Qwen3.5 and LFM2. Baseline
is `32d1582b` on `claude/model-catalog-search-ux-9f83b3`; the implementation is
on `claude/catalog-config-normalizer`. The Gemma 3 image wrapper, external
Qwen3.8 profiles, LLaMBA and ModernBERT are out of scope and unchanged.

## Problem, controls and history

A crawl of the first two catalogue pages of every family found 97 rows that a
family planner rejected only over `config.json` contents: publisher
annotations, sampling defaults or a newer Transformers serialization of the
same setting. A prototype normalizer applied to copies of those configurations
let 63 rows plan against their real tensor headers.

The production pass is one module, `config_normalization.py`, applied on the
line where each in-scope planner reads its configuration. Header admission and
conversion therefore share it. Controls set before the live run: model-free
tests for each rule and its refusal cases, no mutation of the input, a per
family planner test showing a previously rejected configuration now plans with
the same plan as the clean fixture on both paths while an unknown key is still
refused, and one live pass over the 97 rows expecting at least 63 to plan.

Differences from the prototype: rules are scoped to the planners that read the
affected key; `rope_parameters` is not flattened for Qwen3.5 (which requires
it), Gemma 3 or Gemma 4, nor for Qwen3, Llama and LFM2 (which validate the
plain form themselves), nor beside a non-null `rope_scaling`;
`sliding_window_size` is compared only with an explicit integer
`sliding_window` (the Gemma planner has no default); LFM2's
`tie_word_embeddings` is kept when `tie_embedding` is `false`; malformed
annotations are refused instead of dropped.

## Decision and limits

Adopted on the branch for review. The outcome of the live pass is in the
canonical record. Admission still checks tensor storage only; duplicate tied
tensor byte equality remains a full-conversion check. No row was converted,
loaded or executed, so native loading, numerical correctness, cache, image and
MTP qualification of every newly admitted artifact remain separate and
unverified. The attention dropout rewrite is a judgement call isolated behind
`ZERO_INACTIVE_ATTENTION_DROPOUT`.
