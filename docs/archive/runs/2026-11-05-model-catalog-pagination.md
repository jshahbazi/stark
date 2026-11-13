# Model catalog discovery and pagination — 2026-11-05

Source commit: `5828bf256cc706080265c2cce5f44b574f4a6416`.
Branch: `codex/fix-model-catalog`.

## Cause and changes

The catalog inspected at most 45 candidates and selected at most 30 results.
Its `has_more` hint offered no continuation. The known Lily Q4-XL source was
not a discovery seed; a live Qwen3.8 architecture search placed it beyond the
first 250 results. The app consequently displayed a successful empty catalog.
The existing pinned Qwen3.8 mixed source also lacked an app download route.

Discovery now visits deduplicated known sources followed by the family's Hub
architecture iterators. Each request checks at most 30 candidates and reports
its `offset` and advancing `next_offset`. Rejected candidates consume positions;
empty and duplicate-only pages automatically continue until a new row, the end
of discovery or an error. Load more preserves accumulated rows and installation
choices. Request, family, generation and offset checks fence stale replies.

Lily Q4-XL uses its existing pinned source revision and typed converter checks.
The existing pinned `qwen38-mixed` source uses native preparation with exact
catalog-item matching, source-specific staging and prepared identity validation.
No additional native weight format was implemented.

Gated, deleted and missing sources, plus metadata exceeding the bounded read
contract, can be skipped. Unknown inspection failures leave the page retryable
without advancing past an uncertain source. Configuration and quantization
preflight precede shard-index reads, avoiding large indexes for unsupported
formats. Browsing downloads metadata only. Complete tensor compatibility and
native loading remain installation and runtime checks.

## Validation

- Full converter suite on the source commit: 6,024 passed and 89 skipped in
  234.27 seconds with `HF_HUB_OFFLINE=1` and
  `TOKENIZERS_PARALLELISM=false`, using locked uv dependencies and Python 3.13.13.
- Focused catalog, desktop-browser and native catalog suite: 134 passed.
- Swift controller harness: 128 checks, zero failures and zero skips. Coverage
  includes accumulated catalogs larger than one wire page, duplicate rows,
  automatic continuation, stale replies, retry, cancellation and busy state.
- Full app Swift typecheck passed for `arm64-apple-macosx26.4`.
- Menu-bar builder and window suites: 49 passed.
- Built `stark-template` with locked dependencies and Rust 1.98.1 for local tests.
- Converter Ruff lint and formatting, repository policy and diff checks passed.
- Synthetic native Qwen installation exercised preparation, exact identity,
  managed registration, progress and registry persistence without loading a
  worker or downloading a real model.

Live metadata checks returned the pinned Lily Q4-XL and mixed sources on the
initial page. The page at offset 240 returned
`fabiogreter/Qwen3.8-Flash-Next-lily-q4` at
`28ed3245913fca1bf276b53abe7d87351e6a1d4d`, which was unreachable under the
previous fixed discovery ceiling.

## Installed app

Built and installed a Developer ID signed development app at
`/Applications/Stark.app`, retaining the previously installed generation,
scoring and template executables as build inputs. The builder's model-free
protocol checks and strict signature verification passed. The installed
catalog, desktop, CLI and native-catalog Python files match the source commit;
the manifest's Swift source hash matches the committed source.
The manifest records a dirty checkout because a pre-existing, unrelated
untracked research note remained in the workspace.

Development build: `20261105203448206346`.
Installed manifest SHA-256:
`966919d1284b6f41b42f5a88e4c91443e2c408502bfb0f2d7e28e513595b3213`.

| Executable | Installed SHA-256 |
| --- | --- |
| `stark-generate` | `51f6e0e36448e7def31c8e57722fd122008733476e6bb967abcfdf28f3ae5326` |
| `stark-score` | `f18ce85e90c0f50cbec116cfa000934b0226b10dcbff203f42a485a8ab9f5437` |
| `stark-template` | `b0533992307cdcf80cce0f2ebd63ebc6a2605e810bd88ee9c0c3a9de898e0e0b` |

A live app check confirmed the initial Qwen3.8 catalog shows Lily Q4-XL with
its existing Downloaded marker and the mixed source with an enabled Download
button. One Load more action traversed intermediate pages and added Lily Q4
while retaining both earlier rows. The app was left on that catalog screen.
No real model download, model load or inference was performed in this run.
