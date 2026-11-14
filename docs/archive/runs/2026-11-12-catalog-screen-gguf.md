# Model catalogue GGUF-only listing screen

Date is 2026-11-12. Implementation starts from `0366d566` on
`claude/catalog-screen-gguf`.

## Cause

The family catalogues never offer a GGUF download. A GGUF-only repository
passed the listing screen because `.gguf` counts as a weight file there, so
each one cost a repository request and then became an **Unavailable** row
saying the checkpoint requires a supported Safetensors file or a complete
shard index.

One publisher ships the same placeholder `config.json` (`model_type` set to
`mistral`) in every GGUF repository, so its Qwen3, Llama 3.2, Gemma 3, Phi-3.5,
Yi-Coder and SOLAR exports were listed in the **Mistral** family. In the
baseline below, all 25 of Mistral's rows with that reason were GGUF
repositories, 22 of them from that publisher.

## Implemented behavior

- `_listing_rejects` also drops a listed source whose weight files, ignoring
  adapter files, are all `.gguf`. It uses the listing's file names only, so the
  source uses its position without any request, like the other screens, and
  advances `next_offset` the same way.
- A source that lists Safetensors, PyTorch `.bin` or EXL3 weights beside GGUF
  files is still inspected. A `.bin`-only source is still inspected and keeps
  its stated reason.
- Native families (Flint, GLM) still skip listing screens. Registered, pinned
  and native sources are seeded without a listing and are never screened.
  `gguf_import.py`, the Add from Source support probe and the GLM GGUF
  preparation path do not use this screen and are unchanged.
- A ranking test that used a GGUF-only source to stand for a scored source
  without Safetensors now uses a `.bin`-only source; its expectations are
  unchanged.

## Measurement

Three catalogue pages per family, live Hub metadata, unauthenticated, each run
with a fresh verdict cache. Before is `0366d566`; after is this branch. The two
runs were back to back. "Safetensors reason" counts rows whose reason is the
Safetensors-or-shard-index requirement; "GGUF-named" counts rows whose
repository name contains `gguf`.

| Family | Rows before | Rows after | Unavailable before | Unavailable after | Safetensors reason before | Safetensors reason after | GGUF-named before | GGUF-named after | Offset after page 3, before | Offset after page 3, after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mistral | 58 | 49 | 36 | 19 | 25 | 2 | 25 | 0 | 114 | 194 |
| Llama | 66 | 65 | 29 | 27 | 3 | 1 | 2 | 0 | 120 | 124 |
| Phi-3 | 70 | 69 | 66 | 65 | 3 | 0 | 8 | 0 | 137 | 183 |
| Gemma 2 | 46 | 49 | 25 | 23 | 2 | 0 | 3 | 2 | 121 | 130 |

Downloadable rows went from 22 to 30 for Mistral, 37 to 38 for Llama, 4 to 4
for Phi-3 and 21 to 26 for Gemma 2, because the same three pages reach further
down each family.

Page times in seconds:

| Family | Before | After |
| --- | --- | --- |
| Mistral | 2.0, 1.7, 1.2 | 4.5, 5.7, 9.7 |
| Llama | 1.8, 1.6, 2.5 | 8.5, 6.9, 10.2 |
| Phi-3 | 1.0, 2.2, 1.8 | 3.9, 2.4, 6.3 |
| Gemma 2 | 2.2, 6.6, 7.9 | 6.7, 5.7, 9.2 |

The after run was slower in every family, including Llama, whose listing
offsets barely moved. The screen removes requests rather than adding them, so
the difference is not attributed to this change; Hub latency during the second
of two back-to-back runs is the likely cause, but it was not isolated.

After the change, the remaining rows with the Safetensors reason
(`BioMistral/BioMistral-7B`, `openaccess-ai-collective/tiny-mistral`,
`JackFram/llama-68m`) list only `pytorch_model.bin` weights. The two remaining
GGUF-named Gemma 2 rows also list Safetensors weights and are downloadable BF16
rows. `hecan3d/GemmaDGAF-2B`, whose name does not mention GGUF, lists only GGUF
weights and is now screened.

## Validation

- Converter suite: 7,541 passed, 94 skipped, 0 failed, with the native
  helpers built in `target/release/`. New tests in
  `tests/test_catalog_api_reuse.py` cover a GGUF-only listing screened with no
  request; mixed GGUF and Safetensors, `.bin`-only, and GGUF with `.bin`
  sources still inspected; seeded and unlisted GGUF-only sources still
  inspected with their reason; and GGUF-only screened positions advancing
  `next_offset` like other screened positions.
- `ruff check` and `ruff format --check` with the CI command set: pass.
- Repository policy check: pass.

## Not covered

- The catalogue was not inspected in the app.
- No model was downloaded, installed or run; the change affects catalogue
  discovery only.
- Page times were measured once per side and are not a controlled comparison.
