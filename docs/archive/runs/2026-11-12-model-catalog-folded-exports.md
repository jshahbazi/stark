# Model catalogue folded foreign-format exports

Date is 2026-11-12. Implementation starts from `fb8258c5` on
`claude/catalog-fold-foreign-formats`.

## Cause

GPU-server quantization exports (AWQ, GPTQ, bitsandbytes, compressed-tensors
and similar) are the largest group of rows Stark cannot import. Each was its
own **Unavailable** row. With **Hide Unsupported** checked they only added to
the hidden count; unchecked, they were dead ends beside the model they were
made from, and a search for one found nothing a user could download.

In the earlier crawl of the first two pages of every catalogue family, 149 of
152 such rows carried exactly one `base_model:quantized:<repository>` Hub tag,
and none carried more than one. The other three had no base-model tag.

## Implemented behavior

- `catalog_reasons.declared_original` returns the one repository a row's Hub
  tags declare with `base_model:quantized:`, when it is a valid `owner/name`
  identifier other than the row's own repository. Two different declarations,
  a malformed identifier or a malformed tag list yield none.
- A row classified as a GPU-server export carries that repository as
  `original_repo`. No other row does, including supported MLX exports that
  declare the same relation. The declared repository is never requested, and
  the row's reason, label and availability are unchanged.
- The desktop backend and the app each reject `original_repo` on a
  downloadable item, one naming the item's own repository, and one that is
  not an `owner/name` identifier.
- The app folds an export when its declared original, compared without regard
  to letter case, is a downloadable row among the rows loaded. A folded
  export is not a row and is not counted as hidden. With **Hide Unsupported**
  unchecked the original's group names its exports as links, up to six and a
  count of the rest. A search that matches a folded export's name, repository
  or format, and none of the group's own rows, shows the original with the
  same note in either setting.
- An export whose original is not loaded, or is itself unavailable, remains an
  **Unavailable** row and links to the declared original on Hugging Face.

The declaration is the export publisher's own model-card metadata and is not
verified. It can only move a row that could never be downloaded under a row
that can; it never selects, alters or admits a download.

## Measurement

Three catalogue pages per family, live Hub metadata, unauthenticated, with a
fresh verdict cache. The fold rule applied to the loaded rows is the app's.

| Family | Rows | Unavailable | GPU-server exports | Declare an original | Folded | Original unavailable | Original not loaded | Hidden before | Hidden after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2 | 76 | 41 | 31 | 30 | 30 | 0 | 0 | 41 | 11 |
| Qwen3 | 45 | 23 | 20 | 20 | 20 | 0 | 0 | 23 | 3 |
| Mistral | 58 | 36 | 9 | 7 | 7 | 0 | 0 | 36 | 29 |
| Phi-3 | 70 | 66 | 15 | 13 | 6 | 7 | 0 | 66 | 60 |
| Llama | 66 | 29 | 16 | 13 | 0 | 5 | 8 | 29 | 29 |
| Gemma 2 | 46 | 25 | 7 | 6 | 0 | 6 | 0 | 25 | 25 |
| Total | 361 | 220 | 98 | 89 | 63 | 18 | 8 | 220 | 157 |

The 63 folded exports sit under 26 originals. Llama and Gemma 2 fold nothing
in these pages: every declared original there is a `meta-llama` or `google`
repository that is either listed as unavailable or not among the loaded rows.
Phi-3's seven unfolded exports point at `microsoft/phi-4`,
`microsoft/Phi-4-reasoning-plus` and `microsoft/Phi-3.5-mini-instruct`, which
are listed but unavailable.

## Validation

- Converter suite: 7,539 passed, 94 skipped, 0 failed, including 35 new tests
  in `tests/test_catalog_originals.py`.
- `ruff check` and `ruff format --check` with the CI command set: pass.
- Repository policy check: pass.
- Swift controller harness: 150 checks, 0 failures, including the new
  `catalog-foreign-exports-fold-under-original` case.
- App type-check: pass.

## Not covered

- The folded note and the unfolded row's link were not inspected on screen.
- No model was downloaded, installed or run; this change touches catalogue
  display only.
- Exports without a `base_model:quantized:` tag are not matched by name.
- The original of an unfolded export is not fetched, so an export on an early
  page folds only once the page holding its original has loaded.
