# Model catalogue top-rated ordering

Date is 2026-11-12. Implementation starts from `783aa4a8` on
`claude/model-catalog-search-ux-9f83b3`.

## Cause

A family in **Model Catalog** lists many compatible repositories in Hub
download order, sorted by size. Nothing indicated which model to choose. The
Beginner guide already bundles a whatmodel.sh snapshot of parent-model scores,
but the catalogue did not use it.

## Implemented behavior

- `catalog_ranking.py` attaches a `reference_score` to a browsing row when the
  bundled snapshot scores that repository, a reviewed export binding covers its
  exact revision, or the row is a quantized format whose publisher declares or
  names one scored parent. The policy and its limits are in
  [Beginner recommendation evidence](../../MODEL_RECOMMENDATIONS.md#catalogue-ordering).
- Sources with a reference score are inspected before the rest of the first
  300 listing positions, preferring parents from the RAM list this Mac can use.
  A listed source without Safetensors weights gets no priority.
- The app adds **Top rated (general)** and **Top rated (coding)** to family
  detail and opens on the general order unless a sort was saved. Downloadable
  rows whose weights fit 85% of physical memory less 2 GiB come first, then the
  higher score, then the smaller download. Scored rows show the score as a
  link, with a tooltip naming the scored model; rows over the budget are marked
  as likely too large.
- Native install identity ignores only the `reference_score` field. The
  backend and the app both reject a malformed score record.
- A listing that fails partway again contributes the entries it returned; the
  preceding listing-size change had dropped them.

Guided setup's planning scan is neither reordered nor scored by this change.

## Live observations

Live Hub metadata, unauthenticated, converter environment, 128 GiB host. First
catalogue page per family, rows ordered as the app's general rating does.

| Family | Rows | Scored rows | Page time | First row | Leading rows |
| --- | --- | --- | --- | --- | --- |
| Qwen3 dense | 20 | 20 | 2.0 s | 0.7 s | Qwen3-14B 4-bit, 8-bit and BF16 at 58.1, then Qwen3-32B at 57.4 |
| Llama dense | 22 | 19 | 1.7 s | 0.5 s | MiniCPM5-2B at 63.1, then DeepSeek and Llama 3.1/3.2 exports |
| Qwen3.5 dense | 29 | 27 | 2.6 s | 0.7 s | a reviewed Qwen3.8-27B export at 71.8, then Qwen3.5-27B at 67.6 |
| Mistral dense | 13 | 12 | 2.5 s | 0.5 s | Magistral-Small-2506 at 55.2, then Mistral-Small-24B at 53.9 |

Before this change the Llama and Mistral first pages held download-ordered
sources. With a 16 GiB host the Llama page was led by the same top row and
then by exports of models from that host's RAM list. The table's order was
computed outside the app with an earlier, looser fit rule (weights plus 2 GiB
within physical memory); on a 128 GiB host the listed leading rows fit under
either rule.

Scores are parent-model references from the snapshot captured 2026-11-11. No
export was benchmarked, and no row's availability changed.

## Validation

- New `test_catalog_ranking.py`: 23 tests covering each relationship, refusal
  to score full-precision derivatives and ambiguous names, host-list priority,
  inspection order across pages, unchanged admission, a partially failed
  listing, the unchanged guided scan, backend validation of the score record
  and native install identity.
- Focused catalogue, header, desktop-browser, native catalogue, streaming and
  guided suites: 597 passed and 6 failed; the same 6 fail on `72ad1b09` in this
  checkout.
- Full converter suite without the native `stark-template` build: 6,712 passed,
  93 skipped, 384 failed and 161 errors, with the identical 545 failing and
  erroring tests as `72ad1b09`. A full run with the native build remains to be
  done.
- Swift controller harness: 149 checks, zero failures and zero skips, including
  the rating orders at three memory sizes, coding-only scores, and rejection of
  six malformed score records.
- Full app Swift typecheck passed for `arm64-apple-macosx26.4`.
- Converter Ruff lint and format checks and the repository policy check passed.
- No model weights were downloaded, loaded or executed. The app was not rebuilt
  or installed, so the catalogue window, score badges and tooltips were not
  seen on screen.
