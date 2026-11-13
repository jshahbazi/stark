# Model-family converter correctness, 2026-11-10

Attempt `20261110-CONVERT-01` covers converter compatibility and normalization
storage on branch `codex/model-family-correctness`, based on main
`746e482f`. It was registered retrospectively after the initial implementation
and first model-free test slice. This record does not establish full-model
numerical qualification, model quality, performance or installed behavior.

## Fixed criteria and scope

The component gate requires the three retained safe dense configurations to
plan successfully with synthetic tensor dimensions, each admitted semantic or
metadata mutation to remain validated, all implemented catalog inline sampling
profiles to resolve at their exact digest, and unquantized HF/MLX Qwen3.5 norm
exports to produce identical effective gains with the offset applied once.
RNJ and Polaris must stay unavailable until their unsupported YaRN semantics
can be represented and qualified. No model payload is downloaded in this work.

The conversion changes preserve tensor values and selected output precision.
Qwen norms remain widened to F32, recurrent parameters retain the prior F32
policy, and kernel, activation, KV/cache and sampler implementations are
unchanged by this converter patch. Prefix restoration and image-loaded text
performance/MTP are not exercised; these require each resulting artifact and
remain pending. RNJ, Polaris, DeepSeek, Nanbeige and Reka's retained configs
declare causal text architectures and no vision tower. This is source evidence
for these configurations, not a claim about related image-capable checkpoints.

## Source identities and review

Public config files were resolved through the Hugging Face API on 2026-11-10
and retained unmodified as objects in
`tools/converter/tests/fixtures/catalog_family_configs.json`. The digest below
identifies each original file before JSON fixture reserialization.
The fixture also retains each repository's declared model license: Apache 2.0
for Nanbeige, Reka, RNJ and Polaris; MIT for DeepSeek. The fixture SHA-256 is
`8747069967d739d7f701fe0a69629989cd94883bde0e552af79e0a948f4f6dc0`.

| Source | Immutable revision | Original config SHA-256 | Disposition |
| --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` | `37bd455e9679d2959536270fed49d25cc7c290a64f6e52abb97c71345a9cee41` | Admit boolean `use_mrope:false`; true stays unsupported |
| `Nanbeige/Nanbeige4.1-3B` | `9c4555c37921c0982af8fffeee1e0a6e6b953c4f` | `33afb0ed9fc41e2df70b3883faf17a404e9f5f25a9aa2364f39a9d228e02d8b3` | Admit validated zero `embd_pdrop` / `resid_pdrop` annotations |
| `RekaAI/reka-flash-3.1` | `dcf8d6ee3219ca177b519cab47777d42864b54bd` | `b912c7cd0942ce5e3e7b400daaded51a44fd8a0b723afde366338eb5390ae192` | Admit zero classifier dropout and typed classifier label metadata under the causal-LM architecture gate |
| `EssentialAI/rnj-1-instruct` | `5ef96d7a82b2bff8009192979432c9bc64e870cd` | `6c36b82027157061b0b86d53ee2c2b57725f19ccf295b3a29ab4445bf55d6c53` | Planned; YaRN `attn_factor:1.0`, `extrapolation_factor:1.0`, plus singular `layer_type` remain unrepresented |
| `POLARIS-Project/Polaris-4B-Preview` | `dcab747284e6f4818d8a871315f63dc1223ee394` | `6b50588478beebf78c1f7f3f33cb367d1ac21db5e8b1a724e23fb0aa46a98d9d` | Planned; YaRN `attn_factor:1.0` is not the existing engine's factor-derived scale |

The source config links are obtained by appending `/resolve/REVISION/config.json`
to each repository's `https://huggingface.co/REPOSITORY` URL. Their adapter
fixtures replace only dimensions and retain the reviewed extra fields. This
establishes config/tensor-planning compatibility on synthetic data, not loading
the real weights or verifying their outputs.

The [MLX-LM sanitizer](https://github.com/ml-explore/mlx-lm/blob/e7461f15c7cbacf630ee2179d1439743eb98787d/mlx_lm/models/qwen3_5.py)
adds one to zero-centered normalization vectors while moving raw convolution
storage from `[C,1,K]` to saved `[C,K,1]`. This applies independently of weight
quantization. The reviewed source file SHA-256 is
`851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6`.
Stark now uses that retained layout marker for unquantized imports, rejects a
lost/ambiguous marker or mixed raw/baked layouts, and retains its packed MLX
gate. Gain tests verify the saved values are unchanged and the effective
normalization offset is applied exactly once. No third-party source text or
model weights were copied into the implementation.

The [pinned Lalamo Qwen profiles](https://github.com/trymirai/lalamo/blob/831accb8252a6b6606c378048caeb44178f1a050/lalamo/model_import/model_specs/qwen.py)
declare Qwen3.5 2B overrides (`top_k:40`, repetition penalty `1.15`, suffix
repetition length `1024`) and 4B overrides (same except repetition penalty
`1.10`). Those serialized default profiles hash to
`1e9d0c7d3febdfaeef2b3c552a742aa9b5431b564c1b818b30c6ba1a3a95b26b`
and `d21abf07fda599a2963cb67a162efc2bdfad13551e62b92c8fcaad58e9d1638d`.
The resolver now admits and rechecks both exact digests. The upstream 2B Q4 row
copies the 0.8B override including temperature `0.8`; the local catalog/exporter
patch selects the shared 2B profile so BF16/Q4/Q8 retain the same source-file
temperature. This is an explicit local compatibility correction, not an
unmodified export of that upstream row. It does not enable checkpoint sampling
defaults in clients.

## Attempt history and validation

The first integrated slice had 243 passes, 93 skips and 13 failures. Six new
sampling tests omitted required EOS metadata; a profile inventory test included
unimplemented catalog rows; six existing template-render tests could not find
the native helper in this fresh worktree. These were test/setup failures, not
evidence that the affected model families worked. The fixture EOS was supplied,
the profile control was scoped to implemented metadata profiles, and unchanged
merged `stark-template` was supplied through PATH from the PR 121 worktree.
The corrected slice passed **256 tests**, with **93 skips**, in 0.78 seconds.
Skipped external-reference artifact comparisons retain their prior opt-in gates.

The corrected command used locked `uv` Python 3.13.13, offline Hugging Face
mode, and disabled tokenizer parallelism. It selected
`test_family_conversion_regressions.py`, `test_dense_adapters.py`,
`test_registry.py`, `test_metadata.py` and `test_reasoning.py`.
The unchanged template helper is used solely for existing source-bound
rendering controls; it does not qualify native generation for this patch.

The broader family slice passed **939 tests** in 31.22 seconds. It selected
`test_qwen35.py`, `test_mlx_qwen35.py`, `test_mlx_qwen35_partitioned.py`,
`test_qwen35_mtp.py`, `test_qwen35_vision.py`, `test_mlx_dense.py`,
`test_model_catalog.py` and the new regression file. A final added mixed-layout
negative control passed in the isolated **24-test** regression file run.
These runs overlap and their counts must not be added.

Ruff formatting and lint pass on the six changed Python files. No source checkpoint was
converted, loaded or generated here. No GPU work, timing study, new binary
installation, image-loaded parity test or MTP qualification was performed.
The locked Python environment uses `tools/converter/uv.lock` SHA-256
`df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5`.
The existing `stark-template` executable SHA-256 is
`dae215a79440338cd8e53313b601ecba575b0da9cf3955783dfb141eedc5cbb1`;
its source is unchanged between the PR 121 validation commit `1a0517d6` and
this branch's base. Runtime test logs remain ignored under
`runtime/converter-review/`.

Granite 3.3's declared `thinking` field and leading reasoning separation are
already handled on this branch's merged PR 121 base. Its native-rendering
false/true controls pass in the corrected slice; no duplicate API patch was
made here. Reasoning formats other than the helper's leading `<think>` block
still require their own reviewed parser and output controls.

## Decision and remaining gates

The safe compatibility changes and norm-storage correction are adopted on this
branch at component scope, awaiting review and exact artifact qualification.
RNJ and Polaris remain rejected by catalog dispatch, with their previous README
support claims removed. Removing those blockers requires source-matched YaRN
attention/extrapolation controls, correct RNJ layer topology, checkpoint/tool
metadata checks, numerical output controls and the normal onboarding matrix.
No inference-speed or quality claim follows from these changes.
