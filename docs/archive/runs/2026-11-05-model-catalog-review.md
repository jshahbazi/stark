# Model catalog review and malformed metadata isolation — 2026-11-05

Runtime source commit: `16cad68e9f21ec08d308f346f7dc3492fea90467`.
Branch: `codex/fix-model-catalog`.
This supplements the [initial catalog validation](2026-11-05-model-catalog-pagination.md).

## Review finding and correction

Review reproduced a page-level availability regression: one malformed but
readable `config.json` or shard index caused the entire page to fail, including
otherwise compatible choices. Retrying the same page could not repair the
source document.

The bounded JSON parser now identifies malformed syntax, duplicate keys,
non-finite literals, invalid Unicode and excessive nesting as a rejected
candidate. Transport, revision and integrity failures retain retryable page
errors. Regression tests cover mixed malformed/valid pages and continuation
past a complete page of malformed candidates.

Cross-review covered the Python rejection boundary, Swift request/family/offset
fencing, automatic continuation, retained installation choices, native source
pins, choice-specific staging and prepared identity validation. No further
blocking defect was identified. Numeric offsets follow the live Hub download
ordering, so changing Hub results can move candidates across page boundaries;
the catalog does not hold a fixed snapshot of the complete Hub listing.

The added app download entry exposes the existing pinned Pipe preparation and
worker path. Its F32 Qwen graph retains `fresh-per-request` and
`prefix_cache.supported: false`; the native model reference now links that
existing disposition. No inference state path or model format changed.

## Verification

- Full converter suite on the runtime source commit: 6,031 passed and 89 skipped
  in 242.49 seconds with `HF_HUB_OFFLINE=1` and
  `TOKENIZERS_PARALLELISM=false`, locked dependencies and Python 3.13.13.
- Focused catalog, desktop-browser and native catalog suite: 141 passed.
- Catalog cross-review independently reran its 93 tests successfully.
- Swift controller harness: 128 checks, zero failures and zero skips, using
  `uv run --project tools/converter --locked --python 3.13.13 python scripts/test_menubar_controller.py`.
- Ruff lint and format checks, repository policy and diff checks passed.
- Packaged isolated Python checks returned two initial Qwen3.8 choices with
  continuation 30 and one later choice at offset 240 with continuation 270.
- Live metadata review also confirmed nonempty initial Qwen3, Qwen3.5 and Llama
  catalogs. Eligibility remains metadata-based; full tensor admission and
  execution require their existing installation/runtime checks.

## Reviewed installed build

Built and installed a Developer ID signed app at `/Applications/Stark.app`.
Development build: `20261105204635627083`.
Installed manifest SHA-256:
`5217fa0a004eb5278f5c4bc3ca078d285fcfd58686424c71e22ee2bfd4266a3c`.

The installed catalog, desktop, CLI and native catalog Python files match the
runtime source commit; its Swift source hash also matches. Strict signature
verification passed before and after installation. The existing generation,
scoring and template executables were reused unchanged from the preceding
installed app; their hashes remain in the initial validation record.

Live UI inspection confirmed the reviewed build displays the pinned Lily Q4-XL
and mixed sources and subsequently adds Lily Q4 while retaining earlier rows.
No real model weights were downloaded, loaded or executed for this review.
