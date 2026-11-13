# MLX affine import for Qwen2 and Mistral dense

Date is 2026-11-12. Implementation starts from `32d1582b` on branch
`claude/dense-mlx-qwen2-mistral`. Attempt `20261112-DENSE-MLX-01` in the
[experiment ledger](../../model-experiments/2026-11-12-dense-mlx-qwen2-mistral.md).

This record covers structural import planning and a code review of native
support. No weights were downloaded, no artifact was converted or loaded, and
no inference ran. None of these import paths is qualified.

## Cause

**Model Catalog** reported Qwen2 and Mistral MLX exports as unavailable with
"This family and mode do not implement the declared quantized representation".
The shared dense planner (`tools/converter/stark_convert/adapters/dense.py`)
enabled MLX affine repacking only for Llama, and the metadata preflight set
`_MLX` in `model_support.py` omitted both families. The repacking helpers in
`compressed.py` do not depend on the family.

## Change

- `adapters/dense.py`: MLX quantization is parsed for every dense family, the
  non-Llama "Quantized checkpoint is outside qualified dense coverage" refusal
  is removed, and packed matrices are recognised for every dense family.
- `model_support.py`: `qwen2` and `mistral` join `_MLX`, so preflight accepts
  a declared uniform MLX 4-bit or 8-bit quantization and the family registry
  lists `mlx-affine-4bit` and `mlx-affine-8bit` for them.

Unchanged: only uniform 4-bit and 8-bit affine matrices are accepted;
five/six-bit widths and per-module overrides stay rejected; Granite, Gemma 2
and Phi keep their refusals; no Rust code changed.

## Native support by code reading

The planner's output for these families uses the same tensor names, specs and
decoder config fields as Llama MLX and Qwen2/Mistral full precision. The engine
paths were read at `32d1582b`; none was executed here.

| Combination | Evidence in `crates/stark-engine` |
| --- | --- |
| MLX matrix with Qwen2 q/k/v biases | The attention block loads `qkvg_projection` through `Linear::new_with_input_rht` with `has_qkvg_biases` (`src/encodable_block/mixer/attention/mod.rs`, lines 126 and 170-177). `Linear::new_mixed_precision` reads the spec from the `weights` subtree and passes the parent tree for biases for `MLXSpec` exactly as for `FullPrecisionSpec` (`src/encodable_block/linear/mod.rs`, lines 141-171). The bias is a separate `biases` leaf of length `output_dim` (`src/encodable_block/linear/matmul.rs`, lines 53-66), applied as the matmul output transform (lines 346-350). The converter writes it at `qkvg_projection.biases`, distinct from the affine planes under `qkvg_projection.weights.*`. Quantized output-bias kernels have existing Metal parity tests: `parity_bf16_gs32_4bit_mlx_with_bias` and `parity_bf16_gemv_qmv_fused_scale_bias`, plus `a8w_mxu_parity_bf16` bias cases including W8 group 64 and W4 group 64 (`unit/backends/common/kernel/matmul/quant_dispatch_test.rs`, lines 191, 229, 614-620). These tests were not run in this attempt. |
| Sliding-window attention with MLX matrices | The ring cache is derived from `sliding_window_size` only (`attention/mod.rs`, line 219); the projection spec does not enter it. |
| Untied MLX `lm_head` | `UntiedEmbeddingConfig` loads `output_embedding` through `UntiedReadout::load` with its own spec (`src/encodable_block/embedding.rs`, lines 142-160), the same path used by untied Llama MLX. |
| Family-keyed runtime logic | No Rust source in `crates/` refers to the `qwen2`, `mistral` or `llama` model types or their architecture names. The engine is driven by the converted decoder config. |
| Prefix reuse | Rewind requires owned KV heads and no ring; snapshots require owned KV heads (`attention/mod.rs`, lines 399-405). Neither reads the weight spec. The generation worker's key binds the loaded model instance (`crates/stark-generate/src/native.rs`, `prefix_cache_key`). |

Native loader qualification records exist for MLX exports of Llama 3.2,
SmolLM3, Qwen3 and Gemma 3 and for the BF16 Qwen2.5-Coder-0.5B and Mistral 7B
v0.3 conversions (`tools/converter/stark_convert/native_qualifications.json`).
They prove loading for their own artifacts and builds only and do not cover any
repository below.

## Live verification

Real `preflight_model(config, mode="text")` and `plan_from_headers(family, ...,
mode="text")` from the branch, against live Hugging Face metadata:
unauthenticated, config, shard index and tensor headers only, 2.5 seconds
between repositories. Every sample uses group size 64, F16 scale planes and
untied embeddings; none activates a sliding window. The header hash is SHA-256
of the sorted list of tensor names, dtypes and shapes.

| Repository | Revision | Bits | Layers | Tensors | MLX specs | Q/K/V bias fusions | Preflight | Plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lmstudio-community/Qwen2.5-Coder-14B-Instruct-MLX-4bit | `4275f1d1fd379c8a5e8cc655c5a57ef03b912a29` | 4 | 48 | 1,255 | 194 | 48 | conditional | pass |
| lmstudio-community/Qwen2.5-Coder-14B-Instruct-MLX-8bit | `6fec0bda9a7de3b76be0395d35fa9045cc9d4c68` | 8 | 48 | 1,255 | 194 | 48 | conditional | pass |
| lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-4bit | `96572f69fd88e9c29dd10cd3c6755dd7b30d73d9` | 4 | 64 | 1,671 | 258 | 64 | conditional | pass |
| lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit | `7bd37ed2a7f007d752b3e0308a229a0fb6f85114` | 8 | 64 | 1,671 | 258 | 64 | conditional | pass |
| lmstudio-community/QwQ-32B-MLX-4bit | `cba06d2ee32d3b02f741208ad8766c3aabaa0bfd` | 4 | 64 | 1,671 | 258 | 64 | conditional | pass |
| lmstudio-community/QwQ-32B-MLX-8bit | `b3dd4970e2d7f8910a47277380d290392bec84d9` | 8 | 64 | 1,671 | 258 | 64 | conditional | pass |
| mlx-community/Qwen2.5-Coder-7B-Instruct-4bit | `019cc73c45c770444708a6dd8690c66243cc5c80` | 4 | 28 | 735 | 114 | 28 | conditional | pass |
| lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit | `84d7d10913033a123d8bb47c065368d62b5041cb` | 4 | 28 | 735 | 114 | 28 | conditional | pass |
| mlx-community/Mistral-7B-Instruct-v0.3-4bit | `a4b8f870474b0eb527f466a03fbc187830d271f5` | 4 | 32 | 743 | 130 | 0 | conditional | pass |

Every plan has zero full-precision matrix specs. Configuration and header
identities:

| Repository | Config SHA-256 | Header SHA-256 |
| --- | --- | --- |
| Qwen2.5-Coder-14B-Instruct-MLX-4bit | `7a2e85de5e0e115e6aaea7cb71091c5833aa29786fc748ef1d7e607e3a3a18f2` | `cc9763fa5803571b7033f1905cf266a1f05a2d0cd91d130279c12269a0f0782e` |
| Qwen2.5-Coder-14B-Instruct-MLX-8bit | `e9de4227306ce17dd19b41e535814f3f0ae8bfe7ce5884192567d51d39b327f7` | `8915a83bbc84cf3929d82629657a72df46ae567c3a4616a8349c000aedc49e04` |
| Qwen2.5-Coder-32B-Instruct-MLX-4bit | `a4054d0840b87c2feabea6fb2014bcd5b672500b2ec4dfec6f63a31d7a6ed785` | `e32a6334f5cf52e675b4afee57db1ef8ec2fdcc8f408a1c6977ed8570efc8d6a` |
| Qwen2.5-Coder-32B-Instruct-MLX-8bit | `34b00acd8242bb7d159d5bb59714b61619e216ae1eaad4f43e46460478369d8c` | `f76a40df600bd70a82509cecb06035caa23c5866929aaad5cb33fdda01b7eb91` |
| QwQ-32B-MLX-4bit | `a190c170905ef270d553f085fcd74edc5fba68bddb79c275d51e61c0d2e0c30f` | `e32a6334f5cf52e675b4afee57db1ef8ec2fdcc8f408a1c6977ed8570efc8d6a` |
| QwQ-32B-MLX-8bit | `c363a51d6d817727cbd8ea7dbefc7a3e0dd353bb936109c5f4ac85db17a19a1d` | `f76a40df600bd70a82509cecb06035caa23c5866929aaad5cb33fdda01b7eb91` |
| mlx-community Qwen2.5-Coder-7B-Instruct-4bit | `08762352ba9fded858f24bdd7b8d2fe61aabe7bce1dfdb663db242d574c3e678` | `7d4a18b8595ba191b7127794ed30fb7a30a9b0c6ad63bb7daf4d1fbe294e2ff6` |
| lmstudio-community Qwen2.5-Coder-7B-Instruct-MLX-4bit | `fb2246ebce082d6ca20473dcc4b1dfee3f55dc18979b2252859e334669c605b7` | `7d4a18b8595ba191b7127794ed30fb7a30a9b0c6ad63bb7daf4d1fbe294e2ff6` |
| Mistral-7B-Instruct-v0.3-4bit | `efc145d2f53d6d6ef2639be0556b4ecf6faac6cf8a7199ad2b7bb5ebcf443bdd` | `a302b5e656aac24490ce86c917a6bce590f472b146ad21258c4b5eea7299ceb2` |

Equal header hashes mean equal tensor inventories, not equal weights.

## Tests

New model-free contracts in `tools/converter/tests/test_mlx_qwen2_mistral.py`
cover both families: 4-bit and 8-bit, group sizes 32 and 64, tied and untied
heads with byte-exact repacking; Qwen2 attention biases fused separately from
the affine bias planes; sliding windows; and refusals for a missing plane, a
wrong packed shape, mismatched plane dtypes, mixed full-precision/MLX fusion,
an extra tensor, absent quantization config, a non-MLX method, disagreeing
declarations, five/six-bit widths, a per-module override, a tied `lm_head`
that differs in any plane or is incomplete, a missing Qwen2 attention bias, and
a quantization declaration without packed matrices. Preflight and
catalogue-admission tests use the existing fake Hub with pinned header ranges.
The Granite refusal test and one preflight example that used Qwen2 as an
unsupported quantized family now use Granite.

The full offline converter suite reported 6,884 passed, 93 skipped, 387 failed
and 161 errors. The failures and errors are exactly the recorded baseline set
for this checkout, which has no native `stark-template` build: no new failure
and none resolved. `ruff check`, `ruff format --check` and the repository
policy check passed. The native Metal tests cited above were not run.

## Dispositions

- Image support: N/A. Text-only `Qwen2ForCausalLM` and `MistralForCausalLM`
  sources with no `vision_config`.
- MTP: N/A. The complete tensor inventory of each sample contains no draft
  head.
- Generation prefix reuse: implemented, artifact qualification pending. The
  dense path's reuse does not depend on weight precision.
- Function tools: reported per artifact from its codec at import; model
  behavior unverified.

## Not verified

- Native loading of any converted Qwen2 or Mistral MLX artifact.
- Numerical agreement with each BF16 parent, and the effect of rounding F16
  coefficient planes to BF16 in the default output.
- Prefill and decode speed, memory and prefix-reuse exactness on these
  artifacts.
- Tied-embedding MLX exports and active sliding windows in real repositories;
  only synthetic fixtures cover them.
- Any repository or revision not listed above.
