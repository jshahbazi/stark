# Phase 4 format inventory

Date: 2026-10-12. Inventory baseline: `2b47515b4a6b78ae7c81f52e9b4f5cff6d3ac685`; catalog SHA-256: `d855e4c02ada48cca0062e7c273d3d7372b67b4ba70379ef3cf275acd9b78a3a`. Counts below describe this source state; subsequent adapter promotions are recorded separately.

This is a read-only inventory of the pinned Lalamo reference at revision `831accb8252a6b6606c378048caeb44178f1a050` (checkout `/tmp/lalamo.9OLD5y`) and Stark's current catalog. It records loader evidence and planning boundaries; it is not a conversion or runtime qualification.

## Scope and source pins

The 110-entry denominator and per-entry fields are the machine-readable catalog in [`tools/converter/stark_convert/catalog.json`](../tools/converter/stark_convert/catalog.json): `entry_count=110`, with four separately counted Stark extensions. The catalog has 54 `conversion=implemented` and 56 `conversion=planned` entries. Its format distribution is 89 entries with `compressed_format=null`, 11 with `{'scheme':'mlx_affine','bits':4}`, and 10 with the same scheme at 8 bits. The complete name/repository/config/status enumeration is intentionally retained in that catalog; this report does not duplicate a second mutable 110-row list.

The pinned reference files used here are:

* `lalamo/model_import/loaders/huggingface.py`, especially lines 235-278, 281-335, and 338-404.
* `lalamo/model_import/model_configs/huggingface/common.py:28-95` (AWQ, GPTQ, and MLX config schemas).
* `lalamo/compressed/__init__.py:1-42` (output matrix/spec families).
* `lalamo/compressed/int.py:241-370`, `microfloat.py:223-340`, `low_rank.py:32-80`, `lloyd_max.py:388-520`, and `hybrid.py:230-300` (native output-spec implementations).

## Input checkpoint schemes observed in the reference loader

### Full precision

`load_linear` dispatches to `load_full_precision` when neither the AWQ probe nor the MLX probe matches (`huggingface.py:245-278`). A tensor named `weight` therefore describes the full-precision input path only when the surrounding model/config and tensor shapes satisfy the loader; `compressed_format=null` in Stark's catalog is not proof that arbitrary dense repositories qualify.

### AutoAWQ

The actual AWQ probe is the presence of `qweight` (`huggingface.py:235-238`), not a repository name. The loader expects AutoAWQ's packed layout: `qweight [in_channels, out_packed]`, `scales [num_groups, out_channels]`, and optional `qzeros` (`huggingface.py:170-192, 291-302`). It infers only 4- or 8-bit packing (`:118-123, 293-299`), unpacks int32 words (`:142-153`), reverses the AutoAWQ 4-bit channel order (`:303-306`), transposes into the native layout, and retains symmetric versus asymmetric zero points (`:308-331`). Fused layers must either all provide `qzeros` or all omit them; mixed fusion is rejected (`:180-186`). This is real AWQ support for the reference loader, but there are no AWQ rows in the current 110-entry catalog.

### GPTQ

The pinned config schema accepts a `GPTQQuantizationConfig` with `bits`, `checkpoint_format`, `desc_act`, `group_size`, `lm_head`, `meta`, `pack_dtype`, `quant_method='gptq'`, and `sym` (`common.py:43-65`). However, the actual generic loader dispatch in `huggingface.py:235-278` has only two tensor-layout probes: `qweight` is treated as AWQ and `scales` as MLX. There is no GPTQ-specific tensor unpacker, no `g_idx` handling, and no GPTQ `zero`/zero-point correction in this loader. Consequently, a config or `qweight` name must not be reported as GPTQ import support. GPTQ is an unsupported input scheme for this inventory until a source-bound implementation demonstrates its packed ordering, group-index semantics, zero-point convention (including any `zero+1` convention), fusion behavior, and malformed-input rejection.

### MLX affine

The MLX probe is the presence of `scales` (`huggingface.py:240-242`), and the packed layout is explicitly `weight [rows, packed_cols]`, `scales [rows, num_groups]`, plus dequantization `biases` (`:338-404`). Bits are inferred from packed width for 1, 4, or 8 (`:373-383`), then an `MLXSpec(bits, group_size, layout)` is constructed (`:392-403`). The model config's MLX tag carries `group_size` and `bits` (`common.py:67-82`). Current catalog rows with non-null `compressed_format` are the 21 implemented MLX affine identities: 11 4-bit and 10 8-bit entries spanning Qwen3, Llama, SmolLM3, and Gemma3 text. Planned MLX-community rows for LFM2 and Qwen3.5 currently have `compressed_format=null`; that null is a catalog/planning gap, not evidence of dense input.

### Other packed/specialized input cases

The loader contains a separate MXFP4 path that validates U8 blocks and E8M0/U8 scales before constructing `MicrofloatSpec` (`huggingface.py:42-75`), and GPT-OSS gate/up row rearrangement (`:78-93`). Qwen gated-attention reordering rejects packed `qweight` because it lacks format-aware output-channel unpacking (`:738-750`). DeltaNet explicitly rejects AWQ (`:983-985`) while carrying a separate full-precision/MLX branch (`:986-1000`). These are architecture- and tensor-contract-specific cases; they do not establish broad AWQ/GPTQ or arbitrary HF coverage.

## Output quantization families in the reference

The compressed package exports `Int`, `Microfloat`, `LowRank`, `LloydMax`, `Hybrid`, and `MLX` matrix/spec families (`compressed/__init__.py:1-42`). These are native output representations selected by model construction and export semantics. They are not names of HF input checkpoint formats.

* `IntSpec` is the native integer scale/zero-point representation (`int.py:241-370`); the AWQ loader maps a qualified AWQ tensor layout into it, but that does not make every `IntSpec` output an AWQ import.
* `MicrofloatSpec` is the native microfloat packed-parameter representation (`microfloat.py:223-340`); the MXFP4 loader is a concrete input mapping, not a claim that all microfloat variants are importable.
* `LowRankSpec` (`low_rank.py:32-80`), `LloydMaxSpec` (`lloyd_max.py:388-520`), and `HybridSpec` (`hybrid.py:230-300`) are native compressed matrix contracts. No corresponding HF input loader is present in the pinned HuggingFace dispatch.
* `MLXSpec` is both a native output spec and the concrete affine packed input path described above (`mlx.py` and `huggingface.py:338-404`). Its overlap must still be qualified per bit width, group size, layout, architecture, and source revision.
* No pinned source file inspected here establishes a D4S4 input loader or a D4S4 output spec. D4S4 remains unsupported/planned until an exact source contract is identified.

## Current catalog coverage by config class

The 110 rows use these repository/config identities; counts are from `catalog.json`:

| Config class | Rows | Implemented | Planned | Non-null format rows |
|---|---:|---:|---:|---:|
| `HFQwen3Config` | 21 | 21 | 0 | 11 |
| `HFLFM2Config` | 19 | 0 | 19 | 0 |
| `HFQwen35Config` | 19 | 0 | 19 | 0 |
| `HFLlamaConfig` | 11 | 11 | 0 | 5 |
| `HFQwen2Config` | 7 | 7 | 0 | 0 |
| `HFGemma3Config` | 6 | 0 | 6 | 0 |
| `HFGemma3TextConfig` | 5 | 5 | 0 | 2 |
| `HFGemma4Config` | 4 | 0 | 4 | 0 |
| `HFGraniteConfig` | 4 | 4 | 0 | 0 |
| `HFSmolLM3Config` | 3 | 3 | 0 | 2 |
| `HFLlambaConfig` | 2 | 0 | 2 | 0 |
| `HFMistralConfig` | 2 | 2 | 0 | 0 |
| `ModernBERTConfig` | 2 | 0 | 2 | 0 |
| Other singletons (`FishAudioConfig`, `HFPhi3Config`, `HFGPTOssConfig`, `HFMuseGlimmerConfig`, `HFBonsaiConfig`) | 5 | 1 | 4 | 0 |

The catalog's exact repository names, including all 110 rows, are the authoritative enumeration. Notable planned identities include all LFM2/LFM2.5 rows, all Qwen3.5/Qwen3.6/Qwen3.8 rows, Gemma4, GPT-OSS, Bonsai 1-bit, and the planned MLX-community rows. Their repository names do not determine tensor format.

## Planning implications

Existing rows can close only when the input scheme and architecture match an existing identity: current MLX affine rows are the closest format precedent; dense `null` rows are suitable only for genuinely full-precision sources after tensor/config inspection. AWQ or GPTQ checkpoints require new format-qualified identities or explicit per-row format metadata rather than silently reusing dense rows. Low-rank, LloydMax, Hybrid, D4S4, and microfloat outputs likewise require new output-contract identities unless a source-bound input mapping is added.

The near-term Qwen3.5 and LFM2 work should therefore classify each source by actual tensors/config and architecture before catalog updates. Qwen3.5 gated attention and LFM2 state-space/conv paths carry additional loader constraints; the pinned loader's packed-Qwen rejection and DeltaNet AWQ rejection are direct reasons to keep those cases separate. The conversion target remains an independently implemented converter with no Lalamo runtime fallback.

No model downloads, inference, benchmarks, tests, credentials, token vectors, weights, or native runtime artifacts were used for this inventory.
