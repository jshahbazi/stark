# Model catalogue unavailable-row reasons

Date is 2026-11-12. Implementation starts from `32d1582b` on
`claude/model-catalog-search-ux-9f83b3`.

## Cause

A crawl of the first two pages of every catalogue family (live Hub metadata)
found 399 of 614 rows unavailable. Many stated reasons were misleading because
an unrelated check fired first, or because one generic sentence hid the
importer's own message:

| Rows | Root cause | Stated reason before this change |
| --- | --- | --- |
| 152 | `quantization_config.quant_method` names a GPU-server format | an MLX option, family, tied-embedding or Gemma 3 image message, with a `4-bit`, `8-bit` or `Unsupported format` label |
| 14 | Gated repository; tensor headers unreadable without approval | "Could not verify the tensor headers. Refresh the catalog to retry." |
| 63 or more | Tensor planner rejection | the generic tensor names, shapes or packing sentence |

The 152 rows by method: compressed-tensors 67, bitsandbytes 40, AWQ 22, GPTQ
7, ModelOpt 6, FP8 5, AutoRound 2, and one each of GLQ, Quark and torchao.

## Implemented behavior

- New `catalog_reasons.py` holds the wording. A config whose
  `quantization_config` names a `quant_method` other than MLX, with no MLX
  `quantization` block, gets the method name as its quantization label (for
  example `AWQ`, `GPTQ`, `bitsandbytes`, `FP8`, `compressed-tensors`,
  `ModelOpt`, `AutoRound`) and the reason "This checkpoint is quantized with
  <method> for GPU inference servers. Stark imports this model's original
  weights or its MLX exports." It replaces any preflight message, and no
  tensor header is requested. It is applied only where the existing
  import-format check already rejects the config, so availability cannot
  change.
- A repository the Hub marks as gated (`auto` or `manual`) whose tensor header
  read fails now says "This model needs access approval on Hugging Face. Sign
  in to Hugging Face, accept the model's terms on its page, then refresh the
  catalog." The verdict cache never stores it, like the unverified-header
  reason. Non-gated header failures and the 4 MiB header limit keep their
  wording.
- `ConversionError` and `UnsupportedConversion` raised by the header inventory
  or the tensor planner now show their own message, with a final period, cut
  at a character boundary to the backend's 240-byte reason limit. Messages
  with control characters, subclasses and other exception types keep the
  generic sentence. An audit of every `raise` of `ConversionError`,
  `UnsupportedConversion`, `ValueError`, `TypeError`, `KeyError`,
  `OverflowError` or `LayoutError` in `stark_convert` found 1,079 constant
  messages and 40 formatted or variable ones; the formatted ones reachable
  from the planner interpolate only config field names, Qwen3.8 field labels
  and Qwen3.5 vision tensor names built from code constants, never file
  paths, tensor contents or request data.

Guided setup matches only on `downloadable`; no code matched on the old
reason strings or labels other than the verdict cache's unverified check,
which now also covers the access reason.

## Offline check against the crawl

Rebuilding each crawled row's quantization blocks from the recorded summary,
the new rule classifies all 152 foreign rows (none missed, by any method), no
other row, and no downloadable row. All 14 gated rows had a header access
failure and the unverified reason, and now get the access reason. No non-gated
row had a header failure. The longest foreign reason is 143 bytes.

## Live observations

Live Hub metadata, unauthenticated, first catalogue page per family. The base
module and the changed module ran against the same memoized API answers.

| Family | Rows | Downloadable before / after | Reason changes |
| --- | --- | --- | --- |
| Qwen2 | 26 | 12 / 12 | 8 of 14 "declared quantized representation" rows now name AWQ (4), bitsandbytes (2) or GPTQ (2) |
| Llama | 22 | 13 / 13 | 7 "unimplemented options" rows now name compressed-tensors (4), bitsandbytes (2) or AWQ (1); 2 unverified-header rows now ask for access approval |
| Gemma 2 | 10 | 0 / 0 | 3 unverified-header rows ask for access approval; 3 generic layout rows now say "Gemma config options are outside qualified coverage."; 2 of 4 quantized-representation rows name AWQ and bitsandbytes |

On every page the row set and each row's `downloadable` value were identical
before and after. The remaining "declared quantized representation" rows are
MLX affine exports of families without MLX import support.

## Validation

- New `test_catalog_reasons.py`: 50 tests covering ten foreign methods (reason,
  label, no header request), precedence over a preflight message, nested
  `text_config`, unknown method names, five MLX configurations that are not
  foreign, the gated access reason and its recheck through the verdict cache,
  granted access, the header budget, unchanged non-gated failures, a planner
  fault on a gated source, planner and header message pass-through, the byte
  limit, fallback for unexpected types and an unchanged downloadable set.
- `test_catalog_tensor_headers.py` now expects the Qwen3.5 vision planner's own
  message for a missing or misshapen vision tensor.
- Full converter suite without the native `stark-template` build: 6,870
  passed, 93 skipped, 387 failed and 161 errors, with the identical failing and
  erroring tests as the base.
- Converter Ruff lint and format checks passed.
- No model weights were downloaded, loaded or executed. The app was not
  rebuilt, so the reasons were not seen on screen.
