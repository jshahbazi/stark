# Model catalogue Hub API reuse, verdict cache and streamed rows

Date is 2026-11-12. Implementation starts from `72ad1b09` on
`claude/model-catalog-search-ux-9f83b3`.

## Cause

Opening a family in **Model Catalog** took several seconds before any row
appeared, and every visit repeated the work. One page inspected 30 candidates.
Each candidate cost a repository request, a size request and a download per
metadata file, and two ranged reads per weight shard through a Hub redirect.
The page was published only when all 30 were done, and leaving the family
discarded the result.

## Implemented behavior

- The Hub listing is requested with each repository's head revision, task tags
  and file names. A source the listing already rules out (another task, no
  model configuration, no weight file) uses its position without a request, so
  a page passes up to 500 such positions and still inspects at most 30.
- Listings are requested at about the size a page needs and grow only when
  more positions are required.
- Tensor headers of an indexed checkpoint come from one Hub API call
  (`safetensors-metadata`). The reply is used only when the Hub labels it with
  the pinned revision and each header passes the ranged-read record checks
  against the pinned file sizes. Otherwise, and for single-file checkpoints,
  the pinned files are read with exact byte ranges as before; the header body
  reuses the storage location resolved by the length prefix, without Hub
  credentials.
- Metadata files are bounded by the sizes in the pinned file inventory instead
  of a separate size request. A missing configuration and an unsupported
  architecture reported by the Hub for a head revision are rejected before any
  file is fetched.
- Verdicts are stored under `catalog-cache` in the app data folder, bound to
  repository, revision and the installed converter sources: 30 days, or one
  hour for access-dependent outcomes. Unverified verdicts are not stored. A
  registered source's head is trusted for ten minutes after confirmation.
- The catalogue child prints each checked row as a JSON line before its
  report. The backend validates and publishes those rows as `model_catalog`
  events marked `partial`, and the app shows them while the page stays pending.

Guided setup's planning scan does not use the verdict cache or streaming; it
does share the listing screen, the Hub header call and the inventory sizes.

## Measurements

Live Hub metadata, unauthenticated, one run each on the same network within
the same half hour. Wall time for the first catalogue page in the converter's
own environment, not the packaged app. Times vary with network latency; the
request counts do not.

| Page | Before | After, first visit | First row | After, repeat visit |
| --- | --- | --- | --- | --- |
| Qwen3.5, empty Hugging Face cache | 18.4 s | 11.6 s | 1.8 s | 1.4 s |
| Qwen3 | 7.5 s | 3.4 s | 0.6 s | 0.5 s |
| Mistral | 6.8 s | 3.1 s | 1.1 s | 0.4 s |
| Qwen3, second page | 3.7 s, 7 rows over 30 positions | 3.3 s, 10 rows over 55 positions | 0.9 s | 0.5 s |

Before the change a repeat visit cost the same as the "Before" column, except
that the Qwen3.5 page took 10.8 s once its metadata files were on disk.

Requests for the Qwen3 first page: 134 ranged header reads became 19 Hub API
header calls and 2 ranged reads. A repeat visit within ten minutes makes
listing requests only. The packaged catalogue command with streaming and the
cache returned 22 rows as 22 streamed lines plus the report in 0.73 s on a
repeat visit.

Known cost: each page still walks the listing from the start, and a listing
reply now carries file names (about 1.6 kB per repository instead of 0.15 kB).
Pages cover more positions, but a page deep in a family downloads a larger
listing than before.

## Validation

- Focused catalogue, header, desktop-browser, native catalogue, streaming and
  guided discovery suites: 457 passed and 6 failed. The same 6 (two external
  Qwen import, one native Qwen install and three GLM preparation tests) fail on
  the starting commit in this checkout; none exercise this change.
- Full converter suite without the native `stark-template` build that
  `CONTRIBUTING.md` requires first: 6,689 passed, 93 skipped, 384 failed and
  161 errors, against 6,663 passed with the identical 545 failing and erroring
  tests on the starting commit. This run preceded the final listing-size and
  head-trust adjustments; the focused suites above were rerun after them. A
  full run with the native build remains to be done.
- New `test_catalog_api_reuse.py`: 25 tests covering cache hits, revision and
  converter changes, expiry, unverified verdicts, damaged entries, listing
  screening and page positions, Hub header use with five fallback cases,
  credential-free storage reads, streamed rows through the CLI and backend,
  and stale-request fencing.
- Swift controller harness: 148 checks, zero failures and zero skips, including
  partial rows, retry after a failed page and stale partial replies.
- Full app Swift typecheck passed for `arm64-apple-macosx26.4`.
- Converter Ruff lint and format checks passed.
- No model weights were downloaded, loaded or executed. The app was not rebuilt
  or installed, so the catalogue window itself was not exercised.
